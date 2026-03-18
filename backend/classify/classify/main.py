"""Classify FastAPI service — orchestrates NER → NEL and exposes classify/batch endpoints."""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from classify.batch_store import (
    complete_batch,
    create_batch,
    get_batch,
    update_batch,
)
from classify.config import (
    CLASSIFIER_VERSION,
    MAX_BATCH_SIZE,
    MAX_TEXT_LENGTH,
    NEL_API_URL,
    NER_API_URL,
)
from classify.user_config import get_user_config
from shared.job_text import build_input_text, compute_hash

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
log = logging.getLogger("classify-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Tabiya Classify API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response models ---

class ClassifyOptions(BaseModel):
    extract_entities: Optional[List[str]] = None
    top_k: int = Field(5, ge=1, le=50)
    min_similarity: float = Field(0.0, ge=0.0, le=1.0)


class ClassifyRequest(BaseModel):
    text: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    options: Optional[ClassifyOptions] = None


class BatchJob(BaseModel):
    job_id: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None


class BatchRequest(BaseModel):
    jobs: List[BatchJob] = Field(..., min_length=1)
    options: Optional[ClassifyOptions] = None


# --- Core classify logic ---

async def _classify_text(
    input_text: str,
    options: Optional[ClassifyOptions] = None,
    http_client: Optional[httpx.AsyncClient] = None,
) -> dict:
    """Call NER then NEL and merge results."""
    opts = options or ClassifyOptions()
    start = time.time()

    ner_payload: Dict[str, Any] = {"text": input_text}
    if opts.extract_entities:
        ner_payload["entity_types"] = opts.extract_entities

    client = http_client or httpx.AsyncClient(timeout=60.0)
    try:
        ner_resp = await client.post(f"{NER_API_URL}/v1/ner", json=ner_payload)
        ner_resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"NER service error: {e}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"NER service unreachable: {e}")

    ner_data = ner_resp.json()
    ner_entities = ner_data.get("entities", [])

    linkable_types = {"occupation", "skill", "qualification"}
    nel_input = [
        {"text": e["surface_form"], "entity_type": e["entity_type"]}
        for e in ner_entities
        if e["entity_type"] in linkable_types
    ]

    linked_map = {}
    nel_metadata = {}
    if nel_input:
        try:
            nel_resp = await client.post(
                f"{NEL_API_URL}/v1/nel",
                json={
                    "entities": nel_input,
                    "options": {"top_k": opts.top_k, "min_similarity": opts.min_similarity},
                },
            )
            nel_resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"NEL service error: {e}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"NEL service unreachable: {e}")

        nel_data = nel_resp.json()
        nel_metadata = nel_data.get("metadata", {})
        for item in nel_data.get("linked_entities", []):
            key = (item["input_text"], item["entity_type"])
            linked_map[key] = item["matches"]

    merged_entities = []
    entity_counts: Dict[str, int] = {}

    for entity in ner_entities:
        etype = entity["entity_type"]
        entity_counts[etype] = entity_counts.get(etype, 0) + 1
        merged = {
            "entity_type": etype,
            "surface_form": entity["surface_form"],
            "span": entity["span"],
        }
        key = (entity["surface_form"], etype)
        if key in linked_map:
            merged["linked_entities"] = linked_map[key]
        merged_entities.append(merged)

    if http_client is None:
        await client.aclose()

    processing_time = round((time.time() - start) * 1000, 1)
    return {
        "classification": {
            "entities": merged_entities,
            "entity_counts": entity_counts,
        },
        "metadata": {
            "classifier_version": CLASSIFIER_VERSION,
            "model_name": ner_data.get("metadata", {}).get("model_name", "unknown"),
            "linker_model": nel_metadata.get("linker_model", "unknown"),
            "processing_time_ms": processing_time,
            "input_text_hash": compute_hash(input_text),
        },
    }


# --- Endpoints ---

@app.post("/v1/classify")
async def classify(req: ClassifyRequest, x_api_key: Optional[str] = Header(None)):
    input_text = build_input_text(req.model_dump(), allow_text_field=True)
    if not input_text:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'title'+'description'")

    if len(input_text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Text exceeds maximum length ({MAX_TEXT_LENGTH} chars)",
        )

    log.info("Classify request: %d chars", len(input_text))
    result = await _classify_text(input_text, req.options)
    entity_count = sum(result["classification"]["entity_counts"].values())
    log.info(
        "Classify done: %d entities in %.1fms",
        entity_count,
        result["metadata"]["processing_time_ms"],
    )
    return result


@app.post("/v1/classify/batch", status_code=202)
async def submit_batch(req: BatchRequest, x_api_key: Optional[str] = Header(None)):
    if len(req.jobs) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch too large ({len(req.jobs)} jobs). Maximum is {MAX_BATCH_SIZE}.",
        )

    batch_id = str(uuid.uuid4())[:8]
    await create_batch(batch_id, len(req.jobs))

    asyncio.create_task(_process_batch(batch_id, req.jobs, req.options))

    log.info("Batch %s submitted: %d jobs", batch_id, len(req.jobs))
    return {"batch_id": batch_id, "total": len(req.jobs), "status": "processing"}


async def _process_batch(
    batch_id: str, jobs: List[BatchJob], options: Optional[ClassifyOptions]
) -> None:
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, job in enumerate(jobs):
            job_id = job.job_id or f"job_{i}"
            input_text = build_input_text(job.model_dump(), allow_text_field=True)

            if not input_text:
                result = {
                    "job_id": job_id,
                    "status": "error",
                    "error": "No classifiable text found",
                }
            elif len(input_text) > MAX_TEXT_LENGTH:
                result = {
                    "job_id": job_id,
                    "status": "error",
                    "error": f"Text exceeds {MAX_TEXT_LENGTH} char limit",
                }
            else:
                try:
                    classify_result = await _classify_text(input_text, options, client)
                    result = {"job_id": job_id, "status": "completed", **classify_result}
                except Exception as e:
                    log.error("[batch-%s] Job %s failed: %s", batch_id, job_id, e)
                    result = {"job_id": job_id, "status": "error", "error": str(e)}

            await update_batch(batch_id, i + 1, result)

    await complete_batch(batch_id)


@app.get("/v1/batch/{batch_id}/status")
async def batch_status(batch_id: str):
    batch = await get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    return {
        "batch_id": batch_id,
        "status": batch["status"],
        "total": batch["total"],
        "processed": batch["processed"],
    }


@app.get("/v1/batch/{batch_id}/results")
async def batch_results(batch_id: str):
    batch = await get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    return {
        "batch_id": batch_id,
        "status": batch["status"],
        "total": batch["total"],
        "processed": batch["processed"],
        "results": batch["results"],
    }


@app.get("/v1/health")
async def health():
    ner_ok = False
    nel_ok = False

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{NER_API_URL}/v1/health")
            ner_ok = r.status_code == 200
        except Exception:
            pass
        try:
            r = await client.get(f"{NEL_API_URL}/v1/health")
            nel_ok = r.status_code == 200
        except Exception:
            pass

    overall = "healthy" if (ner_ok and nel_ok) else "degraded"
    return {
        "status": overall,
        "service": "classify-api",
        "dependencies": {
            "ner_api": "healthy" if ner_ok else "unavailable",
            "nel_api": "healthy" if nel_ok else "unavailable",
        },
    }


@app.get("/v1/version")
async def version():
    return {"service": "classify-api", "version": CLASSIFIER_VERSION}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("classify.main:app", host="0.0.0.0", port=5001, reload=False)

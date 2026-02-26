"""
Classify API Server — POST /v1/classify + batch endpoints
Orchestrator that calls NER API then NEL API and merges the results.
This is the primary endpoint for the scraper pipeline.

Endpoints:
    POST /v1/classify            — classify a single job
    POST /v1/classify/batch      — submit a batch of jobs
    GET  /v1/batch/<id>/status   — poll batch progress
    GET  /v1/batch/<id>/results  — retrieve batch results

Requires ner_server.py (port 5002) and nel_server.py (port 5003) to be running.

Run: uvicorn app.server.classify_server:app --host 0.0.0.0 --port 5001
"""

import sys
import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import httpx

from app.server.common import setup_logging, add_common_middleware
from util.job_text import build_input_text, compute_hash

setup_logging()
log = logging.getLogger("classify-api")

NER_API_URL = os.getenv("NER_API_URL", "http://localhost:5002")
NEL_API_URL = os.getenv("NEL_API_URL", "http://localhost:5003")
CLASSIFIER_VERSION = "1.0.0"

MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "50000"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "500"))

_batches: dict[str, dict] = {}
_http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(timeout=30.0)
    log.info(f"Classify API starting...")
    log.info(f"  NER API: {NER_API_URL}")
    log.info(f"  NEL API: {NEL_API_URL}")
    log.info(f"  Max text length: {MAX_TEXT_LENGTH}")
    log.info(f"  Max batch size: {MAX_BATCH_SIZE}")
    yield
    await _http_client.aclose()


app = FastAPI(
    title="Classify API",
    version=CLASSIFIER_VERSION,
    lifespan=lifespan,
)

add_common_middleware(app)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ClassifyOptions(BaseModel):
    extract_entities: Optional[list[str]] = None
    top_k: int = Field(default=5, ge=1)
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)


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
    jobs: list[BatchJob] = Field(..., min_length=1)
    options: Optional[ClassifyOptions] = None


# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------

async def _classify_text(input_text: str, options: Optional[ClassifyOptions] = None, request_id: str = "") -> dict:
    """
    Core classification logic — calls NER then NEL and merges results.
    Uses async httpx for non-blocking downstream calls.
    """
    options = options or ClassifyOptions()
    entity_types = options.extract_entities
    top_k = options.top_k
    min_similarity = options.min_similarity

    start = time.time()
    headers = {"X-Request-ID": request_id} if request_id else {}

    ner_payload: dict[str, Any] = {"text": input_text}
    if entity_types:
        ner_payload["entity_types"] = entity_types

    ner_resp = await _http_client.post(f"{NER_API_URL}/v1/ner", json=ner_payload, headers=headers)
    ner_resp.raise_for_status()
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
        nel_resp = await _http_client.post(
            f"{NEL_API_URL}/v1/nel",
            json={"entities": nel_input, "options": {"top_k": top_k, "min_similarity": min_similarity}},
            headers=headers,
        )
        nel_resp.raise_for_status()
        nel_data = nel_resp.json()
        nel_metadata = nel_data.get("metadata", {})

        for item in nel_data.get("linked_entities", []):
            key = (item["input_text"], item["entity_type"])
            linked_map[key] = item["matches"]

    merged_entities = []
    entity_counts: dict[str, int] = {}

    for entity in ner_entities:
        etype = entity["entity_type"]
        entity_counts[etype] = entity_counts.get(etype, 0) + 1

        merged: dict[str, Any] = {
            "entity_type": etype,
            "surface_form": entity["surface_form"],
            "span": entity["span"],
        }

        key = (entity["surface_form"], etype)
        if key in linked_map:
            merged["linked_entities"] = linked_map[key]

        merged_entities.append(merged)

    processing_time = round((time.time() - start) * 1000, 1)
    input_text_hash = compute_hash(input_text)

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
            "input_text_hash": input_text_hash,
        },
    }


# ---------------------------------------------------------------------------
# Single classify endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/classify")
async def classify(body: ClassifyRequest, request: Request):
    rid = request.state.request_id
    data = body.model_dump()

    input_text = build_input_text(data, allow_text_field=True)
    if not input_text:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'title'+'description'")

    if len(input_text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Text exceeds maximum length ({MAX_TEXT_LENGTH} chars)",
        )

    log.info(f"[{rid}] Classify request: {len(input_text)} chars")

    try:
        result = await _classify_text(input_text, body.options, request_id=rid)
    except httpx.HTTPStatusError as e:
        log.error(f"[{rid}] Downstream error: {e}")
        raise HTTPException(status_code=502, detail=f"Downstream API error: {e}")

    entity_count = sum(result["classification"]["entity_counts"].values())
    log.info(f"[{rid}] Classify done: {entity_count} entities in {result['metadata']['processing_time_ms']}ms")
    return result


# ---------------------------------------------------------------------------
# Batch endpoints
# ---------------------------------------------------------------------------

async def _process_batch(batch_id: str, jobs: list[dict], options: Optional[ClassifyOptions]):
    """Background task: classify each job and update batch state."""
    batch = _batches[batch_id]

    for i, job in enumerate(jobs):
        input_text = build_input_text(job, allow_text_field=True)
        job_id = job.get("job_id", f"job_{i}")

        if not input_text:
            batch["results"].append({
                "job_id": job_id,
                "status": "error",
                "error": "No classifiable text found",
            })
        elif len(input_text) > MAX_TEXT_LENGTH:
            batch["results"].append({
                "job_id": job_id,
                "status": "error",
                "error": f"Text exceeds {MAX_TEXT_LENGTH} char limit",
            })
        else:
            try:
                result = await _classify_text(input_text, options, request_id=f"batch-{batch_id}-{i}")
                batch["results"].append({
                    "job_id": job_id,
                    "status": "completed",
                    **result,
                })
            except Exception as e:
                log.error(f"[batch-{batch_id}] Job {job_id} failed: {e}")
                batch["results"].append({
                    "job_id": job_id,
                    "status": "error",
                    "error": str(e),
                })

        batch["completed"] = i + 1

    batch["status"] = "completed"
    batch["finished_at"] = time.time()
    elapsed = round((batch["finished_at"] - batch["started_at"]) * 1000)
    batch["total_processing_time_ms"] = elapsed
    log.info(f"[batch-{batch_id}] Completed: {batch['total']} jobs in {elapsed}ms")


@app.post("/v1/classify/batch", status_code=202)
async def submit_batch(body: BatchRequest, request: Request, background_tasks: BackgroundTasks):
    rid = request.state.request_id

    if len(body.jobs) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch too large ({len(body.jobs)} jobs). Maximum is {MAX_BATCH_SIZE}.",
        )

    batch_id = str(uuid.uuid4())[:12]

    _batches[batch_id] = {
        "status": "processing",
        "total": len(body.jobs),
        "completed": 0,
        "results": [],
        "started_at": time.time(),
        "finished_at": None,
        "total_processing_time_ms": None,
    }

    log.info(f"[{rid}] Batch {batch_id} submitted: {len(body.jobs)} jobs")

    jobs_as_dicts = [j.model_dump() for j in body.jobs]
    background_tasks.add_task(_process_batch, batch_id, jobs_as_dicts, body.options)

    return {
        "batch_id": batch_id,
        "status": "processing",
        "total_jobs": len(body.jobs),
        "poll_url": f"/v1/batch/{batch_id}/status",
        "results_url": f"/v1/batch/{batch_id}/results",
    }


@app.get("/v1/batch/{batch_id}/status")
async def batch_status(batch_id: str):
    batch = _batches.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")

    resp = {
        "batch_id": batch_id,
        "status": batch["status"],
        "total": batch["total"],
        "completed": batch["completed"],
        "progress_pct": round(batch["completed"] / batch["total"] * 100, 1),
    }
    if batch["total_processing_time_ms"] is not None:
        resp["total_processing_time_ms"] = batch["total_processing_time_ms"]

    return resp


@app.get("/v1/batch/{batch_id}/results")
async def batch_results(batch_id: str):
    batch = _batches.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")

    if batch["status"] != "completed":
        return JSONResponse(
            status_code=202,
            content={
                "error": "Batch still processing",
                "status": batch["status"],
                "completed": batch["completed"],
                "total": batch["total"],
            },
        )

    return {
        "batch_id": batch_id,
        "status": "completed",
        "total": batch["total"],
        "total_processing_time_ms": batch["total_processing_time_ms"],
        "results": batch["results"],
    }


# ---------------------------------------------------------------------------
# Health & version
# ---------------------------------------------------------------------------

@app.get("/v1/health")
async def health():
    ner_ok = False
    nel_ok = False

    try:
        r = await _http_client.get(f"{NER_API_URL}/v1/health", timeout=5.0)
        ner_ok = r.status_code == 200
    except httpx.HTTPError:
        pass

    try:
        r = await _http_client.get(f"{NEL_API_URL}/v1/health", timeout=5.0)
        nel_ok = r.status_code == 200
    except httpx.HTTPError:
        pass

    status = "healthy" if (ner_ok and nel_ok) else "degraded"
    return {
        "status": status,
        "service": "classify-api",
        "dependencies": {
            "ner-api": "healthy" if ner_ok else "unavailable",
            "nel-api": "healthy" if nel_ok else "unavailable",
        },
    }


@app.get("/v1/version")
async def version():
    return {
        "service": "classify-api",
        "version": CLASSIFIER_VERSION,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server.classify_server:app", host="0.0.0.0", port=5001, log_level="info")

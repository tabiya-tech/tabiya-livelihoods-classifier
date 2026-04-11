"""Classify FastAPI service — orchestrates NER → NEL and exposes classify/batch endpoints."""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from classify.batch_store import complete_batch, create_batch, fail_batch, get_batch_for_user, update_batch, ensure_indexes
from classify.config import CLASSIFIER_VERSION, MAX_BATCH_SIZE, MAX_TEXT_LENGTH
from classify.get_classify_service import get_classify_service
from classify.models import BatchJob, BatchRequest, ClassifyOptions, ClassifyRequest, ClassifyResponse
from classify.service import IClassifyService
from classify.user_config import (
    create_api_key_for_uid,
    delete_api_key_for_uid,
    get_api_key_user,
    get_firebase_uid,
    get_usage_for_uid,
    get_user_config_for_uid,
    list_api_keys_for_uid,
    record_usage_event,
    set_user_config_for_uid,
)
from shared.job_text import build_input_text

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
log = logging.getLogger("classify-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_indexes()
    yield


app = FastAPI(title="Tabiya Classify API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auxiliary request models ───────────────────────────────────────────────

class UserConfigUpdate(BaseModel):
    ner_type: str | None = None
    nel_type: str | None = None
    ner_model_name: str | None = None
    nel_model_name: str | None = None
    taxonomy_model_id: str | None = None


class CreateApiKeyRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=100)


# ── Classify endpoints ─────────────────────────────────────────────────────

@app.post("/v1/classify", response_model=ClassifyResponse)
async def classify(
    req: ClassifyRequest,
    user_config: dict = Depends(get_api_key_user),
    service: IClassifyService = Depends(get_classify_service),
):
    input_text = build_input_text(req.model_dump(), allow_text_field=True)
    if not input_text:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'title'+'description'")
    if len(input_text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=413, detail=f"Text exceeds maximum length ({MAX_TEXT_LENGTH} chars)")

    log.info("Classify request: %d chars (user: %s)", len(input_text), user_config["user_id"])

    try:
        result = await service.classify(input_text, req.options)
    except Exception as e:
        log.error("Classify failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))

    await record_usage_event(user_config["user_id"])
    entity_count = sum(result.classification.entity_counts.values())
    log.info("Classify done: %d entities in %.1fms", entity_count, result.metadata.processing_time_ms)
    return result


@app.post("/v1/classify/batch", status_code=202)
async def submit_batch(
    req: BatchRequest,
    user_config: dict = Depends(get_api_key_user),
    service: IClassifyService = Depends(get_classify_service),
):
    if len(req.jobs) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch too large ({len(req.jobs)} jobs). Maximum is {MAX_BATCH_SIZE}.",
        )

    batch_id = str(uuid.uuid4())
    await create_batch(batch_id, user_config["user_id"], len(req.jobs))
    asyncio.create_task(_process_batch(batch_id, req.jobs, req.options, user_config["user_id"], service))

    log.info("Batch %s submitted: %d jobs (user: %s)", batch_id, len(req.jobs), user_config["user_id"])
    return {"batch_id": batch_id, "total": len(req.jobs), "status": "processing"}


async def _process_batch(
    batch_id: str,
    jobs: list[BatchJob],
    options: ClassifyOptions | None,
    user_id: str,
    service: IClassifyService,
) -> None:
    try:
        for i, job in enumerate(jobs):
            job_id = job.job_id or f"job_{i}"
            input_text = build_input_text(job.model_dump(), allow_text_field=True)

            if not input_text:
                result = {"job_id": job_id, "status": "error", "error": "No classifiable text found"}
            elif len(input_text) > MAX_TEXT_LENGTH:
                result = {"job_id": job_id, "status": "error", "error": f"Text exceeds {MAX_TEXT_LENGTH} char limit"}
            else:
                try:
                    classify_result = await service.classify(input_text, options)
                    result = {"job_id": job_id, "status": "completed", **classify_result.model_dump()}
                except Exception as e:
                    log.error("[batch-%s] Job %s failed: %s", batch_id, job_id, e)
                    result = {"job_id": job_id, "status": "error", "error": str(e)}

            await update_batch(batch_id, result)

        await complete_batch(batch_id)
        await record_usage_event(user_id)
    except Exception as e:
        log.error("[batch-%s] Batch processing failed: %s", batch_id, e)
        await fail_batch(batch_id, str(e))


@app.get("/v1/batch/{batch_id}/status")
async def batch_status(batch_id: str, user_config: dict = Depends(get_api_key_user)):
    batch = await get_batch_for_user(batch_id, user_config["user_id"])
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    return {"batch_id": batch_id, "status": batch["status"], "total": batch["total"], "processed": batch["processed"]}


@app.get("/v1/batch/{batch_id}/results")
async def batch_results(batch_id: str, user_config: dict = Depends(get_api_key_user)):
    batch = await get_batch_for_user(batch_id, user_config["user_id"])
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    return {
        "batch_id": batch_id,
        "status": batch["status"],
        "total": batch["total"],
        "processed": batch["processed"],
        "results": batch["results"],
    }


# ── Dashboard endpoints (/v1/user/*) ──────────────────────────────────────

@app.get("/v1/user/config")
async def get_config(uid: str = Depends(get_firebase_uid)):
    config = await get_user_config_for_uid(uid)
    return {k: v for k, v in config.items() if k != "user_id"}


@app.put("/v1/user/config", status_code=204)
async def update_config(body: UserConfigUpdate, uid: str = Depends(get_firebase_uid)):
    await set_user_config_for_uid(uid, body.model_dump(exclude_none=True))


@app.get("/v1/user/api-keys")
async def list_keys(uid: str = Depends(get_firebase_uid)):
    return await list_api_keys_for_uid(uid)


@app.post("/v1/user/api-keys", status_code=201)
async def create_key(body: CreateApiKeyRequest, uid: str = Depends(get_firebase_uid)):
    return await create_api_key_for_uid(uid, body.label)


@app.delete("/v1/user/api-keys/{key_id}", status_code=204)
async def delete_key(key_id: str, uid: str = Depends(get_firebase_uid)):
    found = await delete_api_key_for_uid(uid, key_id)
    if not found:
        raise HTTPException(status_code=404, detail="API key not found")


@app.get("/v1/user/usage")
async def get_usage(uid: str = Depends(get_firebase_uid)):
    return await get_usage_for_uid(uid)


# ── Health / version ───────────────────────────────────────────────────────

@app.get("/v1/health")
async def health():
    import httpx
    from classify.config import NEL_API_URL, NER_API_URL

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

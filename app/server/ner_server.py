"""
NER API Server — POST /v1/ner
Extracts entity spans from job-related text.
Does NOT perform entity linking (that's the NEL API).

Run: uvicorn app.server.ner_server:app --host 0.0.0.0 --port 5002
"""

import sys
import os
import time
import logging
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from app.server.common import setup_logging, add_common_middleware

setup_logging()
log = logging.getLogger("ner-api")

MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "50000"))

ner_model = None
_model_load_error: Optional[str] = None


def _load_model():
    global ner_model, _model_load_error
    if ner_model is None and _model_load_error is None:
        try:
            from inference.ner import NERModel
            model_name = os.getenv("NER_MODEL", "tabiya/roberta-base-job-ner")
            ner_model = NERModel(model_name=model_name)
        except Exception as e:
            _model_load_error = str(e)
            log.error(f"Failed to load NER model: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading NER model on startup...")
    _load_model()
    if ner_model:
        log.info("NER model loaded.")
    else:
        log.error(f"NER model failed to load: {_model_load_error}")
        log.info("Starting server anyway (will return 503 on requests)...")
    yield


app = FastAPI(
    title="NER API",
    version="1.0.0",
    lifespan=lifespan,
)

add_common_middleware(app)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class NERRequest(BaseModel):
    text: str = Field(..., min_length=1)
    entity_types: Optional[list[str]] = None


class SpanOut(BaseModel):
    start: int
    end: int


class EntityOut(BaseModel):
    entity_type: str
    surface_form: str
    span: SpanOut


class NERMetadata(BaseModel):
    model_name: str
    entity_count: int
    processing_time_ms: float


class NERResponse(BaseModel):
    entities: list[EntityOut]
    metadata: NERMetadata


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/ner", response_model=NERResponse)
async def extract_entities(body: NERRequest, request: Request):
    rid = request.state.request_id

    if len(body.text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Text exceeds maximum length ({MAX_TEXT_LENGTH} chars)",
        )

    if ner_model is None:
        raise HTTPException(status_code=503, detail=_model_load_error or "NER model not loaded")

    log.info(f"[{rid}] NER request: {len(body.text)} chars")
    start = time.time()

    try:
        entities = ner_model.extract(body.text)
    except Exception as e:
        log.error(f"[{rid}] NER inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    processing_time = round((time.time() - start) * 1000, 1)

    if body.entity_types:
        allowed = {t.lower() for t in body.entity_types}
        entities = [e for e in entities if e["entity_type"] in allowed]

    log.info(f"[{rid}] NER done: {len(entities)} entities in {processing_time}ms")

    return NERResponse(
        entities=entities,
        metadata=NERMetadata(
            model_name=ner_model.model_name,
            entity_count=len(entities),
            processing_time_ms=processing_time,
        ),
    )


@app.get("/v1/health")
async def health():
    model_ok = ner_model is not None
    status = "healthy" if model_ok else "unavailable"
    resp = {
        "status": status,
        "service": "ner-api",
        "model_loaded": model_ok,
    }
    if model_ok:
        resp["model_name"] = ner_model.model_name
    if _model_load_error:
        resp["error"] = _model_load_error

    if not model_ok:
        return JSONResponse(content=resp, status_code=503)
    return resp


@app.get("/v1/version")
async def version():
    return {
        "service": "ner-api",
        "version": "1.0.0",
        "model": os.getenv("NER_MODEL", "tabiya/roberta-base-job-ner"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server.ner_server:app", host="0.0.0.0", port=5002, log_level="info")

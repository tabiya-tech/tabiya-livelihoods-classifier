"""
NEL API Server — POST /v1/nel
Links entity text to ESCO taxonomy entries using embedding similarity.
Does NOT require NER — accepts pre-extracted entities from any source.
This is the endpoint Compass would call directly for skill-to-ESCO linking.

Run: uvicorn app.server.nel_server:app --host 0.0.0.0 --port 5003
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
from typing import Optional, Any

from app.server.common import setup_logging, add_common_middleware

setup_logging()
log = logging.getLogger("nel-api")

MAX_ENTITIES_PER_REQUEST = int(os.getenv("MAX_ENTITIES_PER_REQUEST", "200"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "50"))

nel_linker = None
_linker_load_error: Optional[str] = None


def _load_linker():
    global nel_linker, _linker_load_error
    if nel_linker is None and _linker_load_error is None:
        try:
            from inference.nel import NELLinker
            model_name = os.getenv("LINKER_MODEL", "all-MiniLM-L6-v2")
            nel_linker = NELLinker(similarity_model=model_name)
        except Exception as e:
            _linker_load_error = str(e)
            log.error(f"Failed to load NEL linker: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading NEL linker on startup...")
    _load_linker()
    if nel_linker:
        log.info("NEL linker loaded.")
    else:
        log.error(f"NEL linker failed to load: {_linker_load_error}")
        log.info("Starting server anyway (will return 503 on requests)...")
    yield


app = FastAPI(
    title="NEL API",
    version="1.0.0",
    lifespan=lifespan,
)

add_common_middleware(app)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class EntityInput(BaseModel):
    text: str = Field(..., min_length=1)
    entity_type: str = Field(..., min_length=1)


class NELOptions(BaseModel):
    top_k: int = Field(default=5, ge=1)
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)


class NELRequest(BaseModel):
    entities: list[EntityInput] = Field(..., min_length=1)
    options: Optional[NELOptions] = None


class NELResponse(BaseModel):
    linked_entities: list[dict[str, Any]]
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/nel", response_model=NELResponse)
async def link_entities(body: NELRequest, request: Request):
    rid = request.state.request_id

    if len(body.entities) > MAX_ENTITIES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Too many entities ({len(body.entities)}). Maximum is {MAX_ENTITIES_PER_REQUEST}.",
        )

    options = body.options or NELOptions()
    top_k = min(options.top_k, MAX_TOP_K)
    min_similarity = options.min_similarity

    if nel_linker is None:
        raise HTTPException(status_code=503, detail=_linker_load_error or "NEL linker not loaded")

    entity_dicts = [e.model_dump() for e in body.entities]

    log.info(f"[{rid}] NEL request: {len(entity_dicts)} entities, top_k={top_k}")
    start = time.time()

    try:
        results = nel_linker.link(entity_dicts, top_k=top_k, min_similarity=min_similarity)
    except Exception as e:
        log.error(f"[{rid}] NEL linking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Entity linking failed: {e}")

    processing_time = round((time.time() - start) * 1000, 1)
    log.info(f"[{rid}] NEL done: {len(results)} linked in {processing_time}ms")

    return NELResponse(
        linked_entities=results,
        metadata={
            "linker_model": nel_linker.similarity_model_name,
            "taxonomy": "esco",
            "processing_time_ms": processing_time,
        },
    )


@app.get("/v1/health")
async def health():
    linker_ok = nel_linker is not None
    status = "healthy" if linker_ok else "unavailable"
    resp = {
        "status": status,
        "service": "nel-api",
        "model_loaded": linker_ok,
    }
    if linker_ok:
        resp["linker_model"] = nel_linker.similarity_model_name
    if _linker_load_error:
        resp["error"] = _linker_load_error

    if not linker_ok:
        return JSONResponse(content=resp, status_code=503)
    return resp


@app.get("/v1/version")
async def version():
    return {
        "service": "nel-api",
        "version": "1.0.0",
        "linker_model": os.getenv("LINKER_MODEL", "all-MiniLM-L6-v2"),
        "taxonomy": "esco",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server.nel_server:app", host="0.0.0.0", port=5003, log_level="info")

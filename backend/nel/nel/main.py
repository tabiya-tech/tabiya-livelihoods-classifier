"""NEL FastAPI service — entity linking to ESCO taxonomy."""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
log = logging.getLogger("nel-api")

LINKER_MODEL = os.getenv("LINKER_MODEL", "all-MiniLM-L6-v2")
NEL_FILES_PATH = os.getenv("NEL_FILES_PATH", None)
MAX_ENTITIES_PER_REQUEST = int(os.getenv("MAX_ENTITIES_PER_REQUEST", "200"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "50"))

nel_linker = None
_linker_load_error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global nel_linker, _linker_load_error
    try:
        from nel.linker import NELLinker
        try:
            nel_linker = NELLinker(similarity_model=LINKER_MODEL, files_path=NEL_FILES_PATH, from_cache=True)
            log.info("NEL linker loaded from cache: %s", LINKER_MODEL)
        except Exception as cache_err:
            log.warning("Cache load failed (%s), recomputing embeddings — this will take a few minutes", cache_err)
            nel_linker = NELLinker(similarity_model=LINKER_MODEL, files_path=NEL_FILES_PATH, from_cache=False)
            log.info("NEL linker loaded (embeddings recomputed): %s", LINKER_MODEL)
    except Exception as e:
        _linker_load_error = str(e)
        log.error("Failed to load NEL linker: %s", e)
    yield


app = FastAPI(title="Tabiya NEL API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response models ---

class EntityInput(BaseModel):
    text: str = Field(..., min_length=1)
    entity_type: str


class NELOptions(BaseModel):
    top_k: Optional[int] = Field(5, ge=1, le=50)
    min_similarity: Optional[float] = Field(0.0, ge=0.0, le=1.0)


class NELRequest(BaseModel):
    entities: List[EntityInput] = Field(..., min_length=1)
    options: Optional[NELOptions] = None


class TaxonomyMatch(BaseModel):
    similarity_score: float
    taxonomy: str
    label: str
    code: Optional[str] = None
    uri: Optional[str] = None
    eqf_level: Optional[str] = None


class LinkedEntity(BaseModel):
    input_text: str
    entity_type: str
    matches: List[TaxonomyMatch]


class NELMetadata(BaseModel):
    linker_model: str
    taxonomy: str
    processing_time_ms: float


class NELResponse(BaseModel):
    linked_entities: List[LinkedEntity]
    metadata: NELMetadata


# --- Endpoints ---

@app.post("/v1/nel", response_model=NELResponse)
async def link_entities(req: NELRequest):
    if len(req.entities) > MAX_ENTITIES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Too many entities ({len(req.entities)}). Maximum is {MAX_ENTITIES_PER_REQUEST}.",
        )

    options = req.options or NELOptions()
    top_k = min(options.top_k, MAX_TOP_K)
    min_similarity = options.min_similarity

    if nel_linker is None:
        raise HTTPException(
            status_code=503, detail=_linker_load_error or "NEL linker not loaded"
        )

    log.info("NEL request: %d entities, top_k=%d", len(req.entities), top_k)
    start = time.time()

    try:
        results = nel_linker.link(
            [e.model_dump() for e in req.entities],
            top_k=top_k,
            min_similarity=min_similarity,
        )
    except Exception as e:
        log.error("NEL linking failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Entity linking failed: {e}")

    processing_time = round((time.time() - start) * 1000, 1)
    log.info("NEL done: %d linked in %.1fms", len(results), processing_time)

    return NELResponse(
        linked_entities=results,
        metadata=NELMetadata(
            linker_model=nel_linker.similarity_model_name,
            taxonomy="esco",
            processing_time_ms=processing_time,
        ),
    )


@app.get("/v1/health")
async def health():
    linker_ok = nel_linker is not None
    resp = {
        "status": "healthy" if linker_ok else "unavailable",
        "service": "nel-api",
        "model_loaded": linker_ok,
    }
    if linker_ok:
        resp["linker_model"] = nel_linker.similarity_model_name
    if _linker_load_error:
        resp["error"] = _linker_load_error
    if not linker_ok:
        raise HTTPException(status_code=503, detail=resp)
    return resp


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("nel.main:app", host="0.0.0.0", port=5003, reload=False)

"""NEL FastAPI service — entity linking to ESCO taxonomy.

One image serves every registered language. The language is an input: callers send
``language`` on the request (``es``, ``AR-es``, …); omitting it uses ``TARGET_LANGUAGE``.
Each language has its own linker — its own data pack and sentence-transformer — built at
startup for ``ENABLED_LANGUAGES`` and lazily on first use for the rest.
"""

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from nel.get_nel_service import get_nel_service
from nel.models import NELRequest, NELResponse
from nel.service import INELService
from shared.languages import (
    LANGUAGE_REGISTRY,
    default_language,
    enabled_languages,
    get_language_config,
    normalise_language,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
log = logging.getLogger("nel-api")

NEL_FILES_PATH = os.getenv("NEL_FILES_PATH", None)
MAX_ENTITIES_PER_REQUEST = int(os.getenv("MAX_ENTITIES_PER_REQUEST", "200"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "50"))

# language code -> NELLinker. Populated at startup for ENABLED_LANGUAGES, then lazily.
nel_linkers: Dict[str, object] = {}
_linker_load_errors: Dict[str, str] = {}
_load_lock = threading.Lock()


def _linker_model_for(language: str) -> str:
    """Sentence-transformer for a language (``LINKER_MODEL_<LANG>`` overrides the config)."""
    return get_language_config(language)["nel_similarity_model"]


def _build_linker(language: str):
    """Load one language's linker, recomputing its embedding cache if it is missing."""
    from nel.linker import NELLinker, resolve_files_path

    model = _linker_model_for(language)
    files_path = NEL_FILES_PATH or resolve_files_path(language)
    try:
        linker = NELLinker(
            similarity_model=model, files_path=files_path, from_cache=True, language=language
        )
        log.info("NEL linker loaded from cache: language=%s model=%s", language, model)
        return linker
    except Exception as cache_err:
        log.warning(
            "Cache load failed for language=%s (%s), recomputing embeddings — this will take a few minutes",
            language,
            cache_err,
        )
        linker = NELLinker(
            similarity_model=model, files_path=files_path, from_cache=False, language=language
        )
        log.info("NEL linker loaded (embeddings recomputed): language=%s model=%s", language, model)
        return linker


def get_linker(language: str):
    """Linker for a language, loading it on first use.

    :raises RuntimeError: if the language's linker cannot be loaded.
    """
    linker = nel_linkers.get(language)
    if linker is not None:
        return linker
    with _load_lock:
        linker = nel_linkers.get(language)
        if linker is not None:
            return linker
        try:
            linker = _build_linker(language)
        except Exception as e:
            _linker_load_errors[language] = str(e)
            log.error("Failed to load NEL linker for language=%s: %s", language, e)
            raise RuntimeError(f"NEL linker for language '{language}' is not available: {e}")
        nel_linkers[language] = linker
        _linker_load_errors.pop(language, None)
        return linker


@asynccontextmanager
async def lifespan(app: FastAPI):
    for language in enabled_languages():
        try:
            get_linker(language)
        except Exception:
            pass  # already logged; the service still starts and can serve other languages
    yield


app = FastAPI(title="Tabiya NEL API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---

@app.post("/v1/nel", response_model=NELResponse)
async def link_entities(req: NELRequest, service: INELService = Depends(get_nel_service)):
    if len(req.entities) > MAX_ENTITIES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Too many entities ({len(req.entities)}). Maximum is {MAX_ENTITIES_PER_REQUEST}.",
        )
    language = normalise_language(req.language)
    try:
        return service.link_entities(
            [e.model_dump() for e in req.entities], req.options, language=language
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.error("NEL linking failed (language=%s): %s", language, e)
        raise HTTPException(status_code=500, detail=f"Entity linking failed: {e}")


@app.get("/v1/health")
async def health():
    loaded = sorted(nel_linkers)
    resp = {
        "status": "healthy" if loaded else "unavailable",
        "service": "nel-api",
        "model_loaded": bool(loaded),
        "default_language": default_language(),
        "supported_languages": list(LANGUAGE_REGISTRY),
        "loaded_languages": loaded,
        "linker_models": {lang: _linker_model_for(lang) for lang in LANGUAGE_REGISTRY},
    }
    if loaded:
        # Kept for backwards compatibility with callers reading a single model name.
        resp["linker_model"] = nel_linkers[loaded[0]].similarity_model_name
    if _linker_load_errors:
        resp["errors"] = dict(_linker_load_errors)
    if not loaded:
        raise HTTPException(status_code=503, detail=resp)
    return resp


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("nel.main:app", host="0.0.0.0", port=5003, reload=False)

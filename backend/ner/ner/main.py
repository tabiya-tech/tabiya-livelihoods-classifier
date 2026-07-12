"""NER FastAPI service — entity extraction from job-related text.

One image serves every registered language. The language is an input: callers send
``language`` on the request; omitting it uses ``TARGET_LANGUAGE``. Each language maps to a
token-classification checkpoint in its ``shared/languages/{lang}_config.py`` (override
with ``NER_MODEL_<LANG>``), loaded at startup for ``ENABLED_LANGUAGES`` and lazily for
the rest.
"""

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Dict

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ner.get_ner_service import get_ner_service
from ner.models import NERRequest, NERResponse
from ner.service import INERService
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
log = logging.getLogger("ner-api")

MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "50000"))

# language code -> NERModel. Populated at startup for ENABLED_LANGUAGES, then lazily.
ner_models: Dict[str, object] = {}
_model_load_errors: Dict[str, str] = {}
_load_lock = threading.Lock()
_nltk_ready = False


def _model_name_for(language: str) -> str:
    """Checkpoint for a language (``NER_MODEL_<LANG>`` overrides the language config)."""
    return get_language_config(language)["ner_model"]


def _ensure_nltk() -> None:
    global _nltk_ready
    if _nltk_ready:
        return
    import nltk

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    _nltk_ready = True


def _build_model(language: str):
    from ner.model import NERModel

    cfg = get_language_config(language)
    model_name = cfg["ner_model"]
    if not cfg.get("ner_model_is_language_specific", True):
        log.warning(
            "Language %r has no NER checkpoint of its own and will be served by %s. "
            "Entity extraction on %s text will be poor — set NER_MODEL_%s to a %s or "
            "multilingual token-classification model.",
            language,
            model_name,
            cfg.get("name", language),
            language.upper(),
            cfg.get("name", language),
        )
    model = NERModel(
        model_name=model_name,
        crf=bool(cfg.get("ner_crf", False)),
        sentence_tokenizer_language=cfg.get("sentence_tokenizer_language", "english"),
        label_map=cfg.get("ner_label_map") or None,
    )
    log.info("NER model loaded: language=%s model=%s", language, model_name)
    return model


def get_model(language: str):
    """Model for a language, loading it on first use.

    :raises RuntimeError: if the model cannot be loaded.
    """
    model = ner_models.get(language)
    if model is not None:
        return model
    with _load_lock:
        model = ner_models.get(language)
        if model is not None:
            return model
        try:
            _ensure_nltk()
            model = _build_model(language)
        except Exception as e:
            _model_load_errors[language] = str(e)
            log.error("Failed to load NER model for language=%s: %s", language, e)
            raise RuntimeError(f"NER model for language '{language}' is not available: {e}")
        ner_models[language] = model
        _model_load_errors.pop(language, None)
        return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    for language in enabled_languages():
        try:
            get_model(language)
        except Exception:
            pass  # already logged; the service still starts and can serve other languages
    yield


app = FastAPI(title="Tabiya NER API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---

@app.post("/v1/ner", response_model=NERResponse)
async def extract_entities(req: NERRequest, service: INERService = Depends(get_ner_service)):
    if len(req.text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Text exceeds maximum length ({MAX_TEXT_LENGTH} chars)",
        )
    language = normalise_language(req.language)
    try:
        # The model forward pass is synchronous, CPU-bound, and GIL-holding.
        # Run it in a worker thread so the event loop stays free to service
        # other requests (health checks, concurrent NER calls) instead of
        # blocking the whole process for the duration of inference.
        return await asyncio.to_thread(
            service.extract_entities, req.text, req.entity_types, language=language
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.error("NER inference failed (language=%s): %s", language, e)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")


@app.get("/v1/health")
async def health():
    loaded = sorted(ner_models)
    resp = {
        "status": "healthy" if loaded else "unavailable",
        "service": "ner-api",
        "model_loaded": bool(loaded),
        "default_language": default_language(),
        "supported_languages": list(LANGUAGE_REGISTRY),
        "loaded_languages": loaded,
        "models": {lang: _model_name_for(lang) for lang in LANGUAGE_REGISTRY},
    }
    if loaded:
        # Kept for backwards compatibility with callers reading a single model name.
        resp["model_name"] = ner_models[loaded[0]].model_name
    if _model_load_errors:
        resp["errors"] = dict(_model_load_errors)
    if not loaded:
        raise HTTPException(status_code=503, detail=resp)
    return resp


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ner.main:app", host="0.0.0.0", port=5002, reload=False)

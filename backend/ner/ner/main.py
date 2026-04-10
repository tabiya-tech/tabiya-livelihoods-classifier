"""NER FastAPI service — entity extraction from job-related text."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ner.get_ner_service import get_ner_service
from ner.models import NERRequest, NERResponse
from ner.service import INERService

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
log = logging.getLogger("ner-api")

NER_MODEL_NAME = os.getenv("NER_MODEL", "tabiya/roberta-base-job-ner")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "50000"))

ner_model = None
_model_load_error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ner_model, _model_load_error
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        from ner.model import NERModel
        ner_model = NERModel(model_name=NER_MODEL_NAME)
        log.info("NER model loaded: %s", NER_MODEL_NAME)
    except Exception as e:
        _model_load_error = str(e)
        log.error("Failed to load NER model: %s", e)
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
    try:
        return service.extract_entities(req.text, req.entity_types)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.error("NER inference failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")


@app.get("/v1/health")
async def health():
    model_ok = ner_model is not None
    resp = {
        "status": "healthy" if model_ok else "unavailable",
        "service": "ner-api",
        "model_loaded": model_ok,
    }
    if model_ok:
        resp["model_name"] = ner_model.model_name
    if _model_load_error:
        resp["error"] = _model_load_error
    if not model_ok:
        raise HTTPException(status_code=503, detail=resp)
    return resp


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ner.main:app", host="0.0.0.0", port=5002, reload=False)

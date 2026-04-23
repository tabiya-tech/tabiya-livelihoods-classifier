from fastapi import APIRouter
from classify_v2.config import CLASSIFIER_VERSION

router = APIRouter(tags=["version"])


@router.get("/v2/classify/health")
async def health():
    return {"status": "healthy", "service": "classify-v2"}


@router.get("/v2/classify/version")
async def version():
    return {"service": "classify-v2", "version": CLASSIFIER_VERSION}

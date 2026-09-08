import asyncio
import logging

from fastapi import APIRouter, Request
from classify_v2.config import CLASSIFIER_VERSION

_logger = logging.getLogger(__name__)

router = APIRouter(tags=["version"])


@router.get("/v2/classify/health")
async def health(request: Request):
    registry = getattr(request.app.state, "plugin_registry", None)
    if registry is not None:
        asyncio.ensure_future(registry.refresh())
    return {"status": "healthy", "service": "classify-v2"}


@router.get("/v2/classify/version")
async def version():
    return {"service": "classify-v2", "version": CLASSIFIER_VERSION}

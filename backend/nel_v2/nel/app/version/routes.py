"""Health and version endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/v2/nel", tags=["meta"])


class HealthResponse(BaseModel):
    status: str = "ok"


class VersionResponse(BaseModel):
    version: str


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@router.get("/version", response_model=VersionResponse)
async def version():
    from nel.config import CLASSIFIER_VERSION
    return VersionResponse(version=CLASSIFIER_VERSION)

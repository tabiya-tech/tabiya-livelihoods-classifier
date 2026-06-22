"""Classify v2 FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from classify_v2.app.api_keys.routes.routes import router as api_keys_router
from classify_v2.app.api_keys.service.gcp_key_manager import GcpKeyManager
from classify_v2.app.classification.routes.routes import router as classify_router
from classify_v2.app.version.routes import router as version_router
from classify_v2.config import (
    CORS_ALLOWED_ORIGINS,
    GCP_API_KEYS_PARENT_LOCATION,
    GCP_API_MANAGED_SERVICE,
    GCP_PROJECT_ID,
    LOG_LEVEL,
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
_logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Construct the GCP API Keys async client once on the main event loop.
    # The async grpc transport captures the running loop at __init__ time;
    # building it lazily in a request handler hits a worker thread where
    # uvloop refuses to surface the loop. We do it here so all routes share
    # a single, correctly-bound client.
    if GCP_PROJECT_ID and GCP_API_MANAGED_SERVICE:
        app.state.gcp_key_manager = GcpKeyManager(
            project_id=GCP_PROJECT_ID,
            managed_service=GCP_API_MANAGED_SERVICE,
            location=GCP_API_KEYS_PARENT_LOCATION,
        )
        _logger.info("GCP key manager initialised")
    else:
        app.state.gcp_key_manager = None
        _logger.warning(
            "GCP_PROJECT_ID and/or GCP_API_MANAGED_SERVICE are unset — "
            "/v2/user/api-keys routes will return 500 until configured",
        )
    yield
    _logger.info("Classify v2 shutting down")


app = FastAPI(title="Classify v2", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(version_router)
app.include_router(classify_router)
app.include_router(api_keys_router)

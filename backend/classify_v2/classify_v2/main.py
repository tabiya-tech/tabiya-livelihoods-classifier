"""Classify v2 FastAPI application."""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from classify_v2.app.api_keys.routes.routes import router as api_keys_router
from classify_v2.app.api_keys.service.gcp_key_manager import GcpKeyManager
from classify_v2.app.classification.routes.routes import router as classify_router
from classify_v2.app.pipelines.plugins_routes.routes import router as plugins_router
from classify_v2.app.pipelines.registry import PluginRegistry, load_catalog
from classify_v2.app.pipelines.repository import PipelineRepository
from classify_v2.app.pipelines.routes.routes import router as pipelines_router
from classify_v2.app.server_dependencies.db_dependencies import ClassifyDBProvider
from classify_v2.app.version.routes import router as version_router
from classify_v2.config import APPLICATION_MONGODB_URI
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

    # Plugin registry: resolve URLs, fetch manifests, keep a periodic refresh.
    # A dedicated AsyncClient is stored on app.state so the registry and the
    # future executor share connection pooling to plugin bundles.
    app.state.plugin_http = httpx.AsyncClient(timeout=5.0)
    registry = PluginRegistry(
        catalog=load_catalog(),
        http_client=app.state.plugin_http,
    )
    await registry.refresh()
    await registry.start_background_refresh()
    app.state.plugin_registry = registry
    loaded_count = sum(1 for plugin in registry.list_manifests() if plugin.manifest is not None)
    _logger.info("Loaded %d plugin manifest(s)", loaded_count)

    # Pipelines collection: ensure indexes at startup. Skipped when the
    # application MongoDB isn't configured (local smoke tests, docs builds).
    if APPLICATION_MONGODB_URI:
        try:
            app_db = await ClassifyDBProvider.get_application_db()
            await PipelineRepository(app_db).ensure_indexes()
            _logger.info("Pipelines collection indexes ensured")
        except Exception:
            _logger.exception("Failed to ensure pipelines indexes")

    yield

    await registry.stop_background_refresh()
    await app.state.plugin_http.aclose()
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
app.include_router(plugins_router)
app.include_router(pipelines_router)

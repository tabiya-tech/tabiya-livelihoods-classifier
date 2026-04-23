"""NEL v2 FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from nel.app.linking.routes.routes import router as linking_router
from nel.app.nel_models.routes.routes import router as nel_models_router
from nel.app.taxonomy_models.routes.routes import router as taxonomy_models_router
from nel.app.version.routes import router as version_router
from nel.config import LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))

# Third-party loggers are too noisy at DEBUG — keep them at WARNING
for _noisy in ("pymongo", "motor", "httpx", "httpcore", "asyncio"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
_logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _logger.info("Starting NEL v2 service …")
    from nel.app.server_dependencies.db_dependencies import ClassifierDBProvider
    from nel.app.embeddings_cache.repository.repository import EmbeddingsCacheRepository

    app_db = await ClassifierDBProvider.get_application_db()
    taxonomy_db = await ClassifierDBProvider.get_taxonomy_db()
    repo = EmbeddingsCacheRepository(app_db=app_db, taxonomy_db=taxonomy_db)
    await repo.ensure_indexes()
    _logger.info("NEL v2 service ready")
    yield
    _logger.info("NEL v2 service shutting down")


app = FastAPI(title="NEL v2", version="2.0.0", lifespan=lifespan)

app.include_router(version_router)
app.include_router(nel_models_router)
app.include_router(taxonomy_models_router)
app.include_router(linking_router)

"""Classify v2 FastAPI application."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from classify_v2.app.classification.routes.routes import router as classify_router
from classify_v2.app.version.routes import router as version_router
from classify_v2.config import CORS_ALLOWED_ORIGINS, LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
_logger = logging.getLogger(__name__)

app = FastAPI(title="Classify v2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(version_router)
app.include_router(classify_router)

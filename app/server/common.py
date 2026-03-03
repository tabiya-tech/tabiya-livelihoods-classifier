"""
Shared FastAPI boilerplate — CORS, request-ID middleware, exception
handler, and logging setup used by all API servers.
"""

import uuid
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def add_common_middleware(app: FastAPI) -> None:
    """Attach CORS, request-ID tracking, and the global exception handler."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = request.headers.get(
            "X-Request-ID", str(uuid.uuid4())[:8]
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", "?")
        log = logging.getLogger(app.title)
        log.exception("[%s] Unhandled error: %s", rid, exc)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "request_id": rid},
        )

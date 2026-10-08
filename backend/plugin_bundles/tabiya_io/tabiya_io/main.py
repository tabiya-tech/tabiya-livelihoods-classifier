"""Tabiya-IO bundle — FastAPI dispatcher over installed plugins.

Structurally identical to the Tabiya-Core bundle. Kept as its own module
so the two bundles can be deployed and scaled independently — IO plugins
are cheap and fast; Core plugins are CPU-heavy.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tabiya_plugin_contracts.adapters.auth import require_identity_token
from tabiya_plugin_contracts.adapters.http import make_http_adapter

from tabiya_io.plugins import INSTALLED_PLUGINS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
log = logging.getLogger("tabiya-io-bundle")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "Tabiya-IO bundle starting with %d installed plugin(s): %s",
        len(INSTALLED_PLUGINS),
        [manifest.plugin_id for manifest, _, _ in INSTALLED_PLUGINS],
    )
    yield


app = FastAPI(title="Tabiya-IO Plugin Bundle", version="0.1.0", lifespan=lifespan)
_cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


for manifest, invoke_fn, health_fn in INSTALLED_PLUGINS:
    router = make_http_adapter(manifest, invoke_fn, health_fn)
    app.include_router(
        router,
        prefix=f"/plugin/{manifest.plugin_id}",
        dependencies=[Depends(require_identity_token)],
    )


@app.get("/health")
async def bundle_health() -> dict:
    return {
        "status": "ok",
        "bundle": "tabiya_io",
        "plugin_count": len(INSTALLED_PLUGINS),
        "installed": [manifest.plugin_id for manifest, _, _ in INSTALLED_PLUGINS],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "tabiya_io.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5011")),
        reload=False,
    )

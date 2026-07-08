"""Tabiya-Core bundle — FastAPI dispatcher over installed plugins.

The bundle's job is to expose `/plugin/{plugin_id}/{manifest,invoke,health}`
for every plugin installed in `plugins/__init__.py::INSTALLED_PLUGINS`.

The routing itself is trivial: iterate, call `make_http_adapter`, mount
the returned router under `/plugin/{plugin_id}`. All contract behaviour
(error mapping, timeout enforcement, contract-version emission) lives in
the shared adapter helper.
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

from tabiya_core.plugins import INSTALLED_PLUGINS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
log = logging.getLogger("tabiya-core-bundle")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "Tabiya-Core bundle starting with %d installed plugin(s): %s",
        len(INSTALLED_PLUGINS),
        [manifest.plugin_id for manifest, _, _ in INSTALLED_PLUGINS],
    )
    yield


app = FastAPI(title="Tabiya-Core Plugin Bundle", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=(os.getenv("CORS_ALLOWED_ORIGINS") or "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount each installed plugin's router under `/plugin/{plugin_id}`.
# The auth dependency is a bundle-wide gate; individual plugin routes do
# not need to declare it. `require_identity_token` no-ops in local mode.
for manifest, invoke_fn, health_fn in INSTALLED_PLUGINS:
    router = make_http_adapter(manifest, invoke_fn, health_fn)
    app.include_router(
        router,
        prefix=f"/plugin/{manifest.plugin_id}",
        dependencies=[Depends(require_identity_token)],
    )


@app.get("/health")
async def bundle_health() -> dict:
    """Bundle-level health for docker-compose / Cloud Run to poll.

    Returns 200 while at least the process is alive. Per-plugin health is
    exposed under `/plugin/{plugin_id}/health`.
    """

    return {
        "status": "ok",
        "bundle": "tabiya_core",
        "plugin_count": len(INSTALLED_PLUGINS),
        "installed": [manifest.plugin_id for manifest, _, _ in INSTALLED_PLUGINS],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "tabiya_core.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5010")),
        reload=False,
    )

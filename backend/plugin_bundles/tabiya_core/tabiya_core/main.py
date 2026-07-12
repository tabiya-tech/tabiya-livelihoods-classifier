"""Tabiya-Core bundle — FastAPI dispatcher over installed plugins.

The bundle's job is to expose `/plugin/{plugin_id}/{manifest,invoke,health}`
for every plugin installed in `plugins/__init__.py::INSTALLED_PLUGINS`.

The routing itself is trivial: iterate, call `make_http_adapter`, mount
the returned router under `/plugin/{plugin_id}`. All contract behaviour
(error mapping, timeout enforcement, contract-version emission) lives in
the shared adapter helper.

The lifespan additionally wires the plugin Cores to their real backends:

- NER: `HttpEntityExtractor` posts to `${NER_API_URL}/v1/ner`.
- NEL: `HttpEntityLinker`  posts to `${NEL_V2_API_URL}/v2/nel`.

If either env var is missing, that plugin is left unwired — the plugin's
health endpoint reports `down` and any invoke returns a `PLUGIN_INTERNAL`
envelope with the "extractor/linker not initialised" message. This keeps
the bundle bootable in constrained test envs (no legacy services running)
while surfacing the misconfiguration through the standard contract.
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
from tabiya_core.plugins.nel import core as nel_core
from tabiya_core.plugins.nel.http_linker import HttpEntityLinker
from tabiya_core.plugins.ner import core as ner_core
from tabiya_core.plugins.ner.http_extractor import HttpEntityExtractor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
log = logging.getLogger("tabiya-core-bundle")


NER_API_URL_ENV = "NER_API_URL"
NEL_V2_API_URL_ENV = "NEL_V2_API_URL"


_ner_http_extractor: HttpEntityExtractor | None = None
_nel_http_linker: HttpEntityLinker | None = None


def _wire_ner_extractor() -> HttpEntityExtractor | None:
    """Instantiate + register the NER HTTP extractor if the env var is set."""

    ner_base_url = os.getenv(NER_API_URL_ENV, "").strip()
    if not ner_base_url:
        log.warning(
            "%s is not set — NER plugin will report health=down and every "
            "invoke will return PLUGIN_INTERNAL until it is configured.",
            NER_API_URL_ENV,
        )
        return None
    extractor = HttpEntityExtractor(base_url=ner_base_url)
    ner_core.set_extractor(extractor)
    log.info("NER plugin wired to %s/v1/ner.", ner_base_url)
    return extractor


def _wire_nel_linker() -> HttpEntityLinker | None:
    """Instantiate + register the NEL HTTP linker if the env var is set."""

    nel_base_url = os.getenv(NEL_V2_API_URL_ENV, "").strip()
    if not nel_base_url:
        log.warning(
            "%s is not set — NEL plugin will report health=down and every "
            "invoke will return PLUGIN_INTERNAL until it is configured.",
            NEL_V2_API_URL_ENV,
        )
        return None
    linker = HttpEntityLinker(base_url=nel_base_url)
    nel_core.set_linker(linker)
    log.info("NEL plugin wired to %s/v2/nel.", nel_base_url)
    return linker


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ner_http_extractor, _nel_http_linker

    log.info(
        "Tabiya-Core bundle starting with %d installed plugin(s): %s",
        len(INSTALLED_PLUGINS),
        [manifest.plugin_id for manifest, _, _ in INSTALLED_PLUGINS],
    )

    _ner_http_extractor = _wire_ner_extractor()
    _nel_http_linker = _wire_nel_linker()

    try:
        yield
    finally:
        if _ner_http_extractor is not None:
            await _ner_http_extractor.close()
        if _nel_http_linker is not None:
            await _nel_http_linker.close()


app = FastAPI(title="Tabiya-Core Plugin Bundle", version="0.1.0", lifespan=lifespan)
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

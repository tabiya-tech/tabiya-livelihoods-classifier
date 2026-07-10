"""Installed plugins for the Tabiya-Core bundle.

NER and NEL both defer their heavy dependencies (transformer, MongoDB,
embeddings) behind injectable interfaces. The bundle registers each
plugin unconditionally; each plugin's health function reports `down`
until the implementation is wired in.

The actual wiring (real transformer + real Mongo-backed linker) is the
responsibility of the deployment layer — a follow-up subtask populates
those at bundle startup. During bundle-level tests we inject fakes.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from tabiya_plugin_contracts import Health, HealthStatus, Manifest

from .ner import MANIFEST as NER_MANIFEST
from .ner import invoke as ner_invoke
from .ner.core import _extractor as _ner_extractor_ref  # noqa: F401 — imported for type hints only
from .ner import core as ner_core
from .nel import MANIFEST as NEL_MANIFEST
from .nel import invoke as nel_invoke
from .nel import core as nel_core
from .language_router import MANIFEST as LANGUAGE_ROUTER_MANIFEST
from .language_router import health as language_router_health
from .language_router import invoke as language_router_invoke


HealthFn = Callable[[], Awaitable[Health]]


async def _ner_health() -> Health:
    if ner_core._extractor is None:
        return Health(status=HealthStatus.DOWN, detail="NER extractor not initialised.")
    return Health(status=HealthStatus.OK)


async def _nel_health() -> Health:
    if nel_core._linker is None:
        return Health(status=HealthStatus.DOWN, detail="NEL linker not initialised.")
    return Health(status=HealthStatus.OK)


INSTALLED_PLUGINS: list[tuple[Manifest, object, Optional[HealthFn]]] = [
    (NER_MANIFEST, ner_invoke, _ner_health),
    (NEL_MANIFEST, nel_invoke, _nel_health),
    # Coming-soon plugin: real manifest, no implementation yet.
    (LANGUAGE_ROUTER_MANIFEST, language_router_invoke, language_router_health),
]

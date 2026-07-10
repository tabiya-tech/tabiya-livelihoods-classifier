"""Language Router plugin (COMING SOON).

Ships a real manifest so it appears in the palette, but has no working
implementation yet: invoke raises UnavailableError and health reports down.
When built, it will detect the input language and route to a language-specific
downstream branch. Branching is out of scope for v1 (single linear chains),
so this stays coming-soon until the pipeline model supports branching.
"""

from __future__ import annotations

from tabiya_plugin_contracts import (
    Context,
    Health,
    HealthStatus,
    Manifest,
    PluginCategory,
    Slot,
    SlotType,
)
from tabiya_plugin_contracts.adapters.http import UnavailableError

MANIFEST = Manifest(
    plugin_id="tabiya.branching.language_router.v1",
    name="Language Router",
    version="0.1.0",
    category=PluginCategory.CORE,
    summary="Detects the input language and routes to a language-specific branch.",
    detail="coming soon — branching pipelines are not supported yet",
    icon="globe",
    input_slot=Slot(type=SlotType.RAW_TEXT),
    output_slot=Slot(type=SlotType.RAW_TEXT),
    config_schema={"type": "object", "properties": {}, "additionalProperties": False},
    timeout_ms=5_000,
    **{"x-tabiya-coming-soon": True},
)


async def invoke(input: object, config: dict, context: Context):
    raise UnavailableError(
        "Language Router is not implemented yet (coming soon).",
        detail={"coming_soon": True},
    )


async def health() -> Health:
    return Health(status=HealthStatus.DOWN, detail="coming_soon")


__all__ = ["MANIFEST", "invoke", "health"]

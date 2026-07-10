"""Stop-word Filter transform plugin (COMING SOON).

Ships a real manifest so it appears in the palette, but has no working
implementation yet. When built, it will strip common stop words from the
incoming text before it reaches NER, reducing noise entities.
"""

from __future__ import annotations

from tabiya_plugin_contracts import Manifest, PluginCategory, Slot, SlotType

from .._coming_soon import coming_soon_health, make_coming_soon_invoke

MANIFEST = Manifest(
    plugin_id="tabiya.transform.stopwords.v1",
    name="Stop-word Filter",
    version="0.1.0",
    category=PluginCategory.TRANSFORM,
    summary="Removes common stop words from the text to cut down on noise entities.",
    detail="coming soon",
    icon="filter",
    input_slot=Slot(type=SlotType.RAW_TEXT),
    output_slot=Slot(type=SlotType.RAW_TEXT),
    config_schema={
        "type": "object",
        "properties": {
            "language": {
                "type": "string",
                "title": "Language",
                "description": "Stop-word list language.",
                "default": "en",
            }
        },
        "additionalProperties": False,
    },
    timeout_ms=5_000,
    **{"x-tabiya-coming-soon": True},
)

invoke = make_coming_soon_invoke("Stop-word Filter")
health = coming_soon_health

__all__ = ["MANIFEST", "invoke", "health"]

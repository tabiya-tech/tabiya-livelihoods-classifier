from __future__ import annotations

from tabiya_plugin_contracts import Manifest, PluginCategory, Slot, SlotType


MANIFEST = Manifest(
    plugin_id="tabiya.source.text.v1",
    name="Text Input",
    version="0.1.0",
    category=PluginCategory.SOURCE,
    summary="Feeds raw text into the pipeline.",
    detail="paste, upload, or title + description",
    icon="text",
    input_slot=Slot(type=SlotType.NONE, cardinality="none"),
    output_slot=Slot(type=SlotType.RAW_TEXT),
    config_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "title": "Text",
                "description": "Body text. Provide this OR title/description, not both.",
            },
            "title": {"type": "string", "title": "Title"},
            "description": {"type": "string", "title": "Description"},
        },
        "additionalProperties": False,
    },
    timeout_ms=5_000,
)

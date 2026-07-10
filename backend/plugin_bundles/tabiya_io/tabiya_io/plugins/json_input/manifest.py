from __future__ import annotations

from tabiya_plugin_contracts import Manifest, PluginCategory, Slot, SlotType


MANIFEST = Manifest(
    plugin_id="tabiya.source.json.v1",
    name="JSON Input",
    version="0.1.0",
    category=PluginCategory.SOURCE,
    summary="Feeds text extracted from a JSON payload into the pipeline.",
    detail="parses a JSON object and reads text from a configurable field",
    icon="docs",
    input_slot=Slot(type=SlotType.NONE, cardinality="none"),
    output_slot=Slot(type=SlotType.RAW_TEXT),
    config_schema={
        "type": "object",
        "properties": {
            "json": {
                "type": "string",
                "title": "JSON payload",
                "description": (
                    "A JSON object as a string, e.g. "
                    '{"description": "Head chef wanted"}.'
                ),
            },
            "text_field": {
                "type": "string",
                "title": "Text field",
                "description": (
                    "Name of the field in the JSON object to read the body "
                    "text from."
                ),
                "default": "text",
            },
        },
        "required": ["json"],
        "additionalProperties": False,
    },
    timeout_ms=5_000,
)

from __future__ import annotations

from tabiya_plugin_contracts import Manifest, PluginCategory, Slot, SlotType


MANIFEST = Manifest(
    plugin_id="tabiya.nel.v1",
    name="Tabiya NEL",
    version="0.1.0",
    category=PluginCategory.CORE,
    summary="Links extracted entities to the ESCO taxonomy.",
    detail="MongoDB Atlas vector search",
    icon="nel",
    input_slot=Slot(type=SlotType.ENTITIES),
    output_slot=Slot(type=SlotType.LINKED_ENTITIES),
    config_schema={
        "type": "object",
        "properties": {
            "top_k": {
                "type": "integer",
                "title": "Top K",
                "minimum": 1,
                "maximum": 50,
                "default": 5,
            },
            "min_similarity": {
                "type": "number",
                "title": "Min similarity",
                "minimum": 0,
                "maximum": 1,
                "default": 0.0,
            },
        },
        "additionalProperties": False,
    },
    timeout_ms=45_000,
)

from __future__ import annotations

from tabiya_plugin_contracts import Manifest, PluginCategory, Slot, SlotType


MANIFEST = Manifest(
    plugin_id="tabiya.sink.results.v1",
    name="Results",
    version="0.1.0",
    category=PluginCategory.SINK,
    summary="Consumes linked entities for downstream display.",
    icon="results",
    input_slot=Slot(type=SlotType.LINKED_ENTITIES),
    output_slot=Slot(type=SlotType.NONE, cardinality="none"),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    timeout_ms=5_000,
)

"""Database Sink plugin (COMING SOON).

Ships a real manifest so it appears in the palette, but has no working
implementation yet. When built, it will persist the linked entities to a
configured database instead of returning them in the response.
"""

from __future__ import annotations

from tabiya_plugin_contracts import Manifest, PluginCategory, Slot, SlotType

from .._coming_soon import coming_soon_health, make_coming_soon_invoke

MANIFEST = Manifest(
    plugin_id="tabiya.sink.database.v1",
    name="Database Sink",
    version="0.1.0",
    category=PluginCategory.SINK,
    summary="Writes the linked entities to a configured database.",
    detail="coming soon",
    icon="download",
    input_slot=Slot(type=SlotType.LINKED_ENTITIES),
    output_slot=Slot(type=SlotType.NONE, cardinality="none"),
    config_schema={
        "type": "object",
        "properties": {
            "connection_uri": {
                "type": "string",
                "title": "Connection URI",
                "description": "Database connection string to write results to.",
            },
            "table": {
                "type": "string",
                "title": "Table / collection",
                "description": "Destination table or collection name.",
            },
        },
        "required": ["connection_uri", "table"],
        "additionalProperties": False,
    },
    timeout_ms=10_000,
    **{"x-tabiya-coming-soon": True},
)

invoke = make_coming_soon_invoke("Database Sink")
health = coming_soon_health

__all__ = ["MANIFEST", "invoke", "health"]

from __future__ import annotations

from tabiya_plugin_contracts import Manifest, PluginCategory, Slot, SlotType


MANIFEST = Manifest(
    plugin_id="tabiya.source.json_entities.v1",
    name="JSON Entities",
    version="0.1.0",
    category=PluginCategory.SOURCE,
    summary="Feeds pre-extracted entities from a JSON payload straight into linking.",
    detail="skip NER — link a JSON list of occupations/skills to the taxonomy",
    icon="docs",
    input_slot=Slot(type=SlotType.NONE, cardinality="none"),
    # Emits Entities so it can feed the NEL stage directly (no NER needed):
    # json_entities → nel → results.
    output_slot=Slot(type=SlotType.ENTITIES),
    config_schema={
        "type": "object",
        "properties": {
            "json": {
                "type": "string",
                "title": "Entities JSON",
                "description": (
                    "A JSON array of entities, e.g. "
                    '[{"surface_form": "head chef", "entity_type": "occupation"}]. '
                    "Each item needs surface_form and entity_type; span is optional."
                ),
            },
            "source_text": {
                "type": "string",
                "title": "Source text (optional)",
                "description": "Optional original text these entities came from.",
            },
        },
        "required": ["json"],
        "additionalProperties": False,
    },
    timeout_ms=5_000,
)

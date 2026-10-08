from __future__ import annotations

from tabiya_plugin_contracts import (
    Manifest,
    PluginCategory,
    Slot,
    SlotType,
)


MANIFEST = Manifest(
    plugin_id="tabiya.ner.v1",
    name="Tabiya NER",
    version="0.1.0",
    category=PluginCategory.CORE,
    summary="Named-entity recognition over job-ad prose.",
    detail="roberta-base-job-ner",
    icon="ner",
    input_slot=Slot(type=SlotType.RAW_TEXT),
    output_slot=Slot(type=SlotType.ENTITIES),
    config_schema={
        "type": "object",
        "properties": {
            "entity_types": {
                "type": "array",
                "title": "Entity types",
                "items": {
                    "type": "string",
                    "enum": ["occupation", "skill", "qualification", "experience", "domain"],
                },
                "description": (
                    "Restrict the extractor's output to these types. Leave "
                    "empty to keep everything the model emits."
                ),
            },
        },
        "additionalProperties": False,
    },
    timeout_ms=30_000,
)

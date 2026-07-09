"""NER plugin manifest.

The x-source pointer for `model_id` currently points at the NEL bundle's
`/v2/nel/models` endpoint because that's where the currently-installed
model list lives; when NER gains its own model registry we'll flip it.
"""

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
            "model_id": {
                "type": "string",
                "title": "Model",
                "default": "tabiya/roberta-base-job-ner",
                # Resolved by classify_v2's options proxy against the NEL v2
                # service (which currently owns the model list). Must NOT point
                # back at /v2/plugins/.../options — that recurses infinitely.
                "x-source": "/v2/nel/models",
            },
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

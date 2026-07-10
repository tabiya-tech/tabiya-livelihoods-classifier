from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator
from tabiya_plugin_contracts import (
    Context,
    Entities,
    Entity,
    EntitySpan,
    NoneSlot,
)
from tabiya_plugin_contracts.adapters.http import (
    ConfigInvalidError,
    jsonable_validation_errors,
)


class JsonEntitiesConfig(BaseModel):
    """A JSON array of entities (as a string) plus an optional source text."""

    json_payload: str = Field(alias="json")
    source_text: str = Field(default="")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _payload_present(self) -> "JsonEntitiesConfig":
        if not self.json_payload or not self.json_payload.strip():
            raise ValueError("Provide a non-empty 'json' array of entities.")
        return self


def _entity_from_item(item: Any, index: int) -> Entity:
    if not isinstance(item, dict):
        raise ValueError(f"Entity at index {index} must be a JSON object.")
    surface_form = item.get("surface_form")
    entity_type = item.get("entity_type")
    if not isinstance(surface_form, str) or not surface_form.strip():
        raise ValueError(
            f"Entity at index {index} needs a non-empty 'surface_form' string."
        )
    if not isinstance(entity_type, str) or not entity_type.strip():
        raise ValueError(
            f"Entity at index {index} needs a non-empty 'entity_type' string."
        )
    # Span is optional for pre-extracted entities — default to a zero span.
    raw_span = item.get("span") or {}
    span = EntitySpan(
        start=int(raw_span.get("start", 0)),
        end=int(raw_span.get("end", 0)),
    )
    return Entity(surface_form=surface_form, entity_type=entity_type, span=span)


def _build_entities(config: JsonEntitiesConfig) -> Entities:
    try:
        parsed = json.loads(config.json_payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"'json' is not valid JSON: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError("'json' must decode to a JSON array of entities.")
    entities = [_entity_from_item(item, index) for index, item in enumerate(parsed)]
    return Entities(entities=entities, source_text=config.source_text)


async def invoke(input: NoneSlot, config: dict, context: Context) -> Entities:
    try:
        parsed_config = JsonEntitiesConfig.model_validate(config or {})
        return _build_entities(parsed_config)
    except ValidationError as exc:
        raise ConfigInvalidError(
            "json_entities config failed validation.",
            detail={"errors": jsonable_validation_errors(exc)},
        ) from exc
    except ValueError as exc:
        raise ConfigInvalidError(
            str(exc),
            detail={"errors": [{"message": str(exc)}]},
        ) from exc


__all__ = ["JsonEntitiesConfig", "invoke"]

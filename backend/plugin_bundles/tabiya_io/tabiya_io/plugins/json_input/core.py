from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError, model_validator
from tabiya_plugin_contracts import Context, NoneSlot, RawText
from tabiya_plugin_contracts.adapters.http import (
    ConfigInvalidError,
    jsonable_validation_errors,
)


class JsonInputConfig(BaseModel):
    """A JSON object (as a string) plus the field to read body text from."""

    json_payload: str = Field(alias="json")
    text_field: str = Field(default="text")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _payload_present(self) -> "JsonInputConfig":
        if not self.json_payload or not self.json_payload.strip():
            raise ValueError("Provide a non-empty 'json' payload.")
        return self


def _extract_text(config: JsonInputConfig) -> str:
    try:
        parsed = json.loads(config.json_payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"'json' is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("'json' must decode to a JSON object.")
    if config.text_field not in parsed:
        raise ValueError(
            f"JSON object has no field '{config.text_field}'. "
            f"Available fields: {sorted(parsed.keys())}."
        )
    value = parsed[config.text_field]
    if not isinstance(value, str):
        raise ValueError(
            f"Field '{config.text_field}' must be a string, got "
            f"{type(value).__name__}."
        )
    if not value.strip():
        raise ValueError(f"Field '{config.text_field}' is empty.")
    return value


async def invoke(input: NoneSlot, config: dict, context: Context) -> RawText:
    try:
        parsed_config = JsonInputConfig.model_validate(config or {})
        text = _extract_text(parsed_config)
    except ValidationError as exc:
        raise ConfigInvalidError(
            "json_input config failed validation.",
            detail={"errors": jsonable_validation_errors(exc)},
        ) from exc
    except ValueError as exc:
        raise ConfigInvalidError(
            str(exc),
            detail={"errors": [{"message": str(exc)}]},
        ) from exc
    return RawText(text=text)


__all__ = ["JsonInputConfig", "invoke"]

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, ValidationError, model_validator
from tabiya_plugin_contracts import Context, NoneSlot, RawText
from tabiya_plugin_contracts.adapters.http import (
    ConfigInvalidError,
    jsonable_validation_errors,
)


class TextInputConfig(BaseModel):
    """One of `text` or (`title` + `description`) must be provided."""

    text: Optional[str] = Field(default=None)
    title: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def _one_shape_required(self) -> "TextInputConfig":
        has_text = bool(self.text and self.text.strip())
        has_title_desc = bool(self.title) or bool(self.description)
        if not has_text and not has_title_desc:
            raise ValueError(
                "Provide either 'text' or ('title' and/or 'description')."
            )
        if has_text and has_title_desc:
            raise ValueError(
                "Provide 'text' OR 'title'/'description', not both."
            )
        return self


def _compose(config: TextInputConfig) -> str:
    if config.text:
        return config.text
    parts: list[str] = []
    if config.title:
        parts.append(config.title.strip())
    if config.description:
        parts.append(config.description.strip())
    # Blank line between title and description so downstream sees paragraphs.
    return "\n\n".join(parts)


async def invoke(input: NoneSlot, config: dict, context: Context) -> RawText:
    try:
        parsed_config = TextInputConfig.model_validate(config or {})
    except ValidationError as exc:
        raise ConfigInvalidError(
            "text_input config failed validation.",
            detail={"errors": jsonable_validation_errors(exc)},
        ) from exc
    return RawText(text=_compose(parsed_config))


__all__ = ["TextInputConfig", "invoke"]

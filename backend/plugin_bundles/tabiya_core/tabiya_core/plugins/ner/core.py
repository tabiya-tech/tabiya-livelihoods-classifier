"""NER plugin Core — pure business logic behind the entity extractor.

The heavy transformer model lives behind `IEntityExtractor` so:
  * Unit tests inject a deterministic fake.
  * The bundle's runtime configuration injects the real transformer-backed
    extractor without the Core function knowing the difference.
  * A future MCP adapter reuses the same `invoke` function verbatim.

The Core's contract is pinned to the shared plugin types:
  * Input: `RawText`
  * Output: `Entities`
  * Config: `NerConfig` (validated inside `invoke`; the adapter passes a
    plain dict per the current contract, so we parse locally).
"""

from __future__ import annotations

from typing import Optional, Protocol

from pydantic import BaseModel, Field, ValidationError
from tabiya_plugin_contracts import (
    Context,
    Entities,
    Entity,
    EntitySpan,
    RawText,
)
from tabiya_plugin_contracts.adapters.http import (
    BadInputError,
    ConfigInvalidError,
    jsonable_validation_errors,
)


_DEFAULT_MODEL_ID = "tabiya/roberta-base-job-ner"


class NerConfig(BaseModel):
    """Typed NER plugin config. Kept minimal in v1."""

    entity_types: Optional[list[str]] = Field(
        default=None,
        description=(
            "If set, keep only entities whose type is in this list. Model can "
            "emit types the request-side taxonomy doesn't recognise (e.g. "
            "'experience', 'domain'); leaving this None returns everything."
        ),
    )


class IEntityExtractor(Protocol):
    async def extract(self, text: str, model_id: str) -> list[Entity]: ...


_extractor: Optional[IEntityExtractor] = None


def set_extractor(extractor: IEntityExtractor) -> None:
    """Register the extractor implementation the plugin will use at invoke-time.

    Called once at bundle startup with the real transformer-backed extractor,
    and by tests to inject a deterministic fake.
    """

    global _extractor
    _extractor = extractor


def get_extractor() -> IEntityExtractor:
    if _extractor is None:
        raise RuntimeError(
            "NER extractor not initialised. Call set_extractor(...) before invoke."
        )
    return _extractor


async def invoke(input: RawText, config: dict, context: Context) -> tuple[Entities, dict]:
    if not input.text.strip():
        raise BadInputError("Field 'text' is required and cannot be empty.")

    try:
        parsed_config = NerConfig.model_validate(config or {})
    except ValidationError as exc:
        raise ConfigInvalidError(
            "NER config failed validation.",
            detail={"errors": jsonable_validation_errors(exc)},
        ) from exc

    extractor = get_extractor()
    all_entities = await extractor.extract(input.text, _DEFAULT_MODEL_ID)

    if parsed_config.entity_types:
        allowed = {label.lower() for label in parsed_config.entity_types}
        filtered = [
            entity for entity in all_entities if entity.entity_type.lower() in allowed
        ]
    else:
        filtered = list(all_entities)

    return Entities(entities=filtered, source_text=input.text), {"model_name": _DEFAULT_MODEL_ID}


__all__ = [
    "IEntityExtractor",
    "NerConfig",
    "get_extractor",
    "invoke",
    "set_extractor",
]


# Re-export Entity + EntitySpan so extractor implementations can construct
# them without importing the contracts package directly if they don't want
# to. Also documents the return shape.
__all__ += ["Entity", "EntitySpan"]

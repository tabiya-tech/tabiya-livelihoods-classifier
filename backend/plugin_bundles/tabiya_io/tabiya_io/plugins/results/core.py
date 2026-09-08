from __future__ import annotations

from pydantic import BaseModel, ValidationError
from tabiya_plugin_contracts import Context, LinkedEntities, NoneSlot
from tabiya_plugin_contracts.adapters.http import (
    ConfigInvalidError,
    jsonable_validation_errors,
)


class ResultsConfig(BaseModel):
    """No config knobs in v1 — reserved shape for future sinks."""

    model_config = {"extra": "forbid"}


async def invoke(input: LinkedEntities, config: dict, context: Context) -> NoneSlot:
    try:
        ResultsConfig.model_validate(config or {})
    except ValidationError as exc:
        raise ConfigInvalidError(
            "results sink config failed validation.",
            detail={"errors": jsonable_validation_errors(exc)},
        ) from exc

    # The sink deliberately does not mutate the payload; the executor is
    # responsible for capturing LinkedEntities before it enters this stage.
    _ = input  # explicit no-op so linters don't flag the unused parameter
    return NoneSlot()


__all__ = ["ResultsConfig", "invoke"]

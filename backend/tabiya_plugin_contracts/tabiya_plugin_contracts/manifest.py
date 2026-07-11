"""Manifest — the static metadata a plugin exposes at `GET /plugin/manifest`.

Capability metadata rides as namespaced `x-tabiya-*` extension fields so
adding one doesn't require a contract-version bump.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .slots import Slot


class PluginCategory(str, Enum):
    SOURCE = "source"
    CORE = "core"
    TRANSFORM = "transform"
    SINK = "sink"


class Manifest(BaseModel):
    """Static metadata returned by `GET /plugin/manifest`.

    The `x-tabiya-*` extension fields are declared as explicit optionals to
    keep the JSON Schema readable; unknown extension fields are also allowed
    so plugin authors can add their own without touching this package.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    plugin_id: str = Field(
        ...,
        description="Dotted, version-suffixed identifier. Example: tabiya.ner.v1",
    )
    name: str
    version: str = Field(..., description="Semver of the plugin implementation itself.")
    category: PluginCategory
    summary: str
    detail: Optional[str] = None
    icon: str = Field(..., description="Key into the frontend's icon map.")
    input_slot: Slot
    output_slot: Slot
    config_schema: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Hand-rolled JSON Schema subset used by the editor's ConfigForm and by "
            "invoke-time config validation. May include the `x-source` extension "
            "field pointing at a plugin options endpoint for dynamic dropdowns."
        ),
    )
    timeout_ms: int = Field(30_000, description="Per-invoke deadline enforced by the orchestrator.")

    x_tabiya_contract_version: Optional[str] = Field(
        default=None,
        alias="x-tabiya-contract-version",
        description="Contract version the plugin was built against. Auto-emitted by the adapter helper.",
    )
    x_tabiya_streams: Optional[bool] = Field(default=None, alias="x-tabiya-streams")
    x_tabiya_idempotent: Optional[bool] = Field(default=None, alias="x-tabiya-idempotent")
    x_tabiya_cancellable: Optional[bool] = Field(default=None, alias="x-tabiya-cancellable")
    x_tabiya_batch_max: Optional[int] = Field(default=None, alias="x-tabiya-batch-max")
    x_tabiya_coming_soon: Optional[bool] = Field(
        default=None,
        alias="x-tabiya-coming-soon",
        description=(
            "When true, the plugin ships a real manifest but has no working "
            "implementation yet. The palette shows it greyed/undroppable and "
            "the validator rejects pipelines that use it."
        ),
    )

"""Persisted shape of a pipeline document.

Kept deliberately dumb — the repository does not enforce contract-level
invariants (source-first, sink-last, slot compatibility). That's the
validator's job in 11.5. Here we only enforce structural well-formedness.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StageDocument(BaseModel):
    """One stage of a pipeline as stored in Mongo.

    `config` is a free-form object validated against the referenced plugin's
    `config_schema` by the validator layer (11.5). Keeping it as `dict` here
    lets the repository stay decoupled from plugin manifests.
    """

    model_config = ConfigDict(extra="forbid")

    plugin_id: str
    config: dict[str, Any] = Field(default_factory=dict)


class PipelineDocument(BaseModel):
    """A user-owned pipeline document.

    Invariants enforced at this layer:
      * `pipeline_id` is unique per user (partial unique index — actually
        enforced only on active docs; we assume callers pass UUIDs).
      * `is_active` is unique per user (partial unique index).
      * `is_readonly=True` means updates/deletes are refused with
        `ReadonlyPipelineError` — used for the seeded Default Tabiya.
    """

    model_config = ConfigDict(extra="forbid")

    pipeline_id: str
    user_id: str
    name: str
    stages: list[StageDocument] = Field(default_factory=list)
    is_active: bool = False
    is_default: bool = False
    is_readonly: bool = False
    created_at: datetime
    updated_at: datetime

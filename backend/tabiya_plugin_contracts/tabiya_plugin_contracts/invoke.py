"""InvokeRequest / InvokeResponse / ErrorEnvelope / Context — the runtime wire types.

The `input` and `output` fields are typed as `dict[str, Any]` at the outer
model level; adapters parse them against the slot model resolved from the
plugin's manifest (`SLOT_MODEL_BY_TYPE[manifest.input_slot.type]`).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    PLUGIN_INTERNAL = "PLUGIN_INTERNAL"
    BAD_INPUT = "BAD_INPUT"
    TIMEOUT = "TIMEOUT"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    CONFIG_INVALID = "CONFIG_INVALID"
    UNAVAILABLE = "UNAVAILABLE"


class Context(BaseModel):
    """Per-invoke context passed to Core functions.

    Adapters populate this from the HTTP request (or from a test fixture in
    the in-process adapter).
    """

    request_id: str
    user_id: Optional[str] = None
    pipeline_id: Optional[str] = None
    stage_index: int = 0
    deadline_ms: int = Field(
        default=30_000,
        description="Wall-clock deadline the plugin SHOULD respect. Adapter enforces a hard timeout on top.",
    )


class InvokeRequest(BaseModel):
    context: Context
    config: dict[str, Any] = Field(default_factory=dict)
    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Slot payload matching manifest.input_slot.type. Adapter parses this against SLOT_MODEL_BY_TYPE.",
    )


class InvokeResponse(BaseModel):
    output: dict[str, Any] = Field(
        default_factory=dict,
        description="Slot payload matching manifest.output_slot.type.",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Free-form per-stage metadata (model_name, processing_time_ms, …).",
    )


class ErrorEnvelope(BaseModel):
    code: ErrorCode
    message: str
    detail: Optional[dict[str, Any]] = None

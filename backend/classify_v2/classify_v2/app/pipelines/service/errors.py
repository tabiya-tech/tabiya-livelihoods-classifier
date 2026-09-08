"""Service-layer errors + typed validation issues.

`ValidationIssue` is a single, addressable problem the frontend can point
its user at. `PipelineValidationError` bundles all issues from a single
validation pass so we return them together — the editor renders one issue
per node/edge in a single 422 response rather than making the user fix
them one at a time.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class IssueCode(str, Enum):
    """Every distinct thing the validator can complain about.

    Values are frontend-facing i18n keys. Adding a new rule adds a new
    enum member; the editor's badge component switches on the code.
    """

    TOO_FEW_STAGES = "too_few_stages"
    UNKNOWN_PLUGIN = "unknown_plugin"
    COMING_SOON_PLUGIN = "coming_soon_plugin"
    UNAVAILABLE_PLUGIN = "unavailable_plugin"
    NOT_A_SOURCE = "not_a_source"
    NOT_A_SINK = "not_a_sink"
    MULTIPLE_SOURCES = "multiple_sources"
    MULTIPLE_SINKS = "multiple_sinks"
    SLOT_MISMATCH = "slot_mismatch"
    STAGE_CONFIG_INVALID = "stage_config_invalid"
    NER_LIMIT_EXCEEDED = "ner_limit_exceeded"
    NEL_LIMIT_EXCEEDED = "nel_limit_exceeded"


class ValidationIssue(BaseModel):
    """A single issue surfaced by `PipelineValidator`.

    `stage_index` is None for pipeline-wide issues (too-few-stages,
    multiple-sources, etc.). `detail` carries structured extras — for
    slot mismatches it's `{previous: SlotType, current: SlotType}`; for
    config errors it's the jsonschema error list.
    """

    code: IssueCode
    message: str
    stage_index: Optional[int] = None
    plugin_id: Optional[str] = None
    detail: Optional[dict[str, Any]] = None


class PipelineServiceError(Exception):
    """Base for pipeline service-layer errors."""


class PipelineValidationError(PipelineServiceError):
    """Validation failed. Carries every issue found in one pass."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        super().__init__(
            f"Pipeline validation failed ({len(issues)} issue(s))."
        )
        self.issues = issues

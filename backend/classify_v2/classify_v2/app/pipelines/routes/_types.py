"""Request + response types for /v2/pipelines routes."""

from __future__ import annotations

from pydantic import BaseModel

from classify_v2.app.pipelines.repository import PipelineDocument, StageDocument
from classify_v2.app.pipelines.service.errors import ValidationIssue


class ListPipelinesResponse(BaseModel):
    pipelines: list[PipelineDocument]


class ValidatePipelineRequest(BaseModel):
    stages: list[StageDocument]


class ValidatePipelineResponse(BaseModel):
    """Standalone validation returns 200 either way — clients render issues
    inline. The routes that actually mutate the store 422 on issues instead.
    """

    valid: bool
    issues: list[ValidationIssue]

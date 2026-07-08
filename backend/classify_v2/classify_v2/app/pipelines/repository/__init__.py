"""Pipelines Mongo repository.

Public surface:
  * `PipelineDocument`, `StageDocument` — persisted shape.
  * `PipelineRepository`, `IPipelineRepository` — CRUD.
  * `ReadonlyPipelineError`, `PipelineNotFoundError` — repository errors.
"""

from ._types import PipelineDocument, StageDocument
from .errors import PipelineNotFoundError, ReadonlyPipelineError
from .repository import (
    ACTIVE_UNIQUE_INDEX_NAME,
    PIPELINES_COLLECTION,
    USER_INDEX_NAME,
    IPipelineRepository,
    PipelineRepository,
)

__all__ = [
    "ACTIVE_UNIQUE_INDEX_NAME",
    "IPipelineRepository",
    "PIPELINES_COLLECTION",
    "PipelineDocument",
    "PipelineNotFoundError",
    "PipelineRepository",
    "ReadonlyPipelineError",
    "StageDocument",
    "USER_INDEX_NAME",
]

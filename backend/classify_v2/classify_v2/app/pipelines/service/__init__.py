"""Pipeline service: validator + service.

Exports:
  * `PipelineValidator`, `ValidationIssue`, `ValidationError` — 11.5 §7 rules.
  * `PipelineService`, `IPipelineService`, `CreatePipelineInput`,
    `UpdatePipelineInput` — the CRUD + validate + activate + clone facade.
  * `DEFAULT_TABIYA_NAME`, `build_default_tabiya_stages()` — the seed helper.
"""

from .errors import (
    PipelineServiceError,
    PipelineValidationError,
    ValidationIssue,
)
from .service import (
    CreatePipelineInput,
    IPipelineService,
    PipelineService,
    UpdatePipelineInput,
)
from .seed import DEFAULT_TABIYA_NAME, build_default_tabiya_stages
from .validator import PipelineValidator

__all__ = [
    "CreatePipelineInput",
    "DEFAULT_TABIYA_NAME",
    "IPipelineService",
    "PipelineService",
    "PipelineServiceError",
    "PipelineValidationError",
    "PipelineValidator",
    "UpdatePipelineInput",
    "ValidationIssue",
    "build_default_tabiya_stages",
]

"""Pipeline service.

Composes the repository (11.4), validator (11.5), and registry (11.2)
into a single facade the routes call into. Every mutation validates
first so the DB never holds a pipeline that would fail at invoke time
(minus the invoke-time re-check the executor still does for post-save
manifest drift — design §7).
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field
from pymongo.errors import DuplicateKeyError

from classify_v2.app.pipelines.registry import PluginRegistry
from classify_v2.app.pipelines.repository import (
    IPipelineRepository,
    PipelineDocument,
    PipelineNotFoundError,
    ReadonlyPipelineError,
    StageDocument,
)
from classify_v2.app.pipelines.service.errors import (
    PipelineServiceError,
    PipelineValidationError,
    ValidationIssue,
)
from classify_v2.app.pipelines.service.seed import (
    DEFAULT_TABIYA_NAME,
    build_default_tabiya_stages,
)
from classify_v2.app.pipelines.service.validator import PipelineValidator

_logger = logging.getLogger(__name__)


class CreatePipelineInput(BaseModel):
    """Payload the create endpoint accepts."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    stages: list[StageDocument] = Field(default_factory=list)


class UpdatePipelineInput(BaseModel):
    """Payload the update endpoint accepts. `pipeline_id` comes from the URL."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    stages: list[StageDocument] = Field(default_factory=list)


class DefaultTabiyaConfig(BaseModel):
    """Values the service uses when seeding Default Tabiya for a user."""

    nel_model_id: str
    taxonomy_model_id: str
    top_k: int = 5
    min_similarity: float = 0.0


class IPipelineService(ABC):
    @abstractmethod
    async def list_for_user(self, user_id: str) -> list[PipelineDocument]: ...

    @abstractmethod
    async def get(self, *, user_id: str, pipeline_id: str) -> PipelineDocument: ...

    @abstractmethod
    async def create(self, *, user_id: str, request: CreatePipelineInput) -> PipelineDocument: ...

    @abstractmethod
    async def update(
        self, *, user_id: str, pipeline_id: str, request: UpdatePipelineInput
    ) -> PipelineDocument: ...

    @abstractmethod
    async def delete(self, *, user_id: str, pipeline_id: str) -> None: ...

    @abstractmethod
    async def activate(self, *, user_id: str, pipeline_id: str) -> PipelineDocument: ...

    @abstractmethod
    async def clone(self, *, user_id: str, pipeline_id: str) -> PipelineDocument: ...

    @abstractmethod
    def validate(self, stages: list[StageDocument]) -> list[ValidationIssue]: ...

    @abstractmethod
    async def ensure_default(
        self, *, user_id: str, default_config: DefaultTabiyaConfig
    ) -> PipelineDocument:
        """Seed a Default Tabiya pipeline for the user if one doesn't exist.

        Returns the (possibly pre-existing) default. Idempotent — safe to
        call on every list/classify request per design §10.
        """


class PipelineService(IPipelineService):
    def __init__(
        self,
        *,
        repository: IPipelineRepository,
        validator: PipelineValidator,
        registry: PluginRegistry,
    ) -> None:
        self._repository = repository
        self._validator = validator
        self._registry = registry

    async def list_for_user(self, user_id: str) -> list[PipelineDocument]:
        return await self._repository.list_for_user(user_id)

    async def get(self, *, user_id: str, pipeline_id: str) -> PipelineDocument:
        doc = await self._repository.get(user_id=user_id, pipeline_id=pipeline_id)
        if doc is None:
            raise PipelineNotFoundError(pipeline_id, user_id)
        return doc

    async def create(
        self, *, user_id: str, request: CreatePipelineInput
    ) -> PipelineDocument:
        issues = self._validator.validate(request.stages)
        if issues:
            raise PipelineValidationError(issues)

        now = datetime.now(timezone.utc)
        doc = PipelineDocument(
            pipeline_id=str(uuid.uuid4()),
            user_id=user_id,
            name=request.name,
            stages=list(request.stages),
            is_active=False,
            is_default=False,
            is_readonly=False,
            created_at=now,
            updated_at=now,
        )
        await self._repository.insert(doc)
        return doc

    async def update(
        self,
        *,
        user_id: str,
        pipeline_id: str,
        request: UpdatePipelineInput,
    ) -> PipelineDocument:
        existing = await self.get(user_id=user_id, pipeline_id=pipeline_id)
        if existing.is_readonly:
            raise ReadonlyPipelineError(pipeline_id)

        issues = self._validator.validate(request.stages)
        if issues:
            raise PipelineValidationError(issues)

        updated_doc = existing.model_copy(
            update={
                "name": request.name,
                "stages": list(request.stages),
            }
        )
        return await self._repository.update(updated_doc)

    async def delete(self, *, user_id: str, pipeline_id: str) -> None:
        await self._repository.delete(user_id=user_id, pipeline_id=pipeline_id)

    async def activate(
        self, *, user_id: str, pipeline_id: str
    ) -> PipelineDocument:
        doc = await self.get(user_id=user_id, pipeline_id=pipeline_id)
        # Re-validate at activate time — a plugin the pipeline references
        # may have been removed since save.
        issues = self._validator.validate(doc.stages)
        if issues:
            raise PipelineValidationError(issues)
        return await self._repository.set_active(
            user_id=user_id, pipeline_id=pipeline_id
        )

    async def clone(
        self, *, user_id: str, pipeline_id: str
    ) -> PipelineDocument:
        source = await self.get(user_id=user_id, pipeline_id=pipeline_id)
        now = datetime.now(timezone.utc)
        clone_doc = PipelineDocument(
            pipeline_id=str(uuid.uuid4()),
            user_id=user_id,
            name=self._clone_name(source.name),
            stages=[stage.model_copy(deep=True) for stage in source.stages],
            is_active=False,
            is_default=False,
            is_readonly=False,
            created_at=now,
            updated_at=now,
        )
        await self._repository.insert(clone_doc)
        return clone_doc

    def validate(self, stages: list[StageDocument]) -> list[ValidationIssue]:
        return self._validator.validate(stages)

    async def ensure_default(
        self, *, user_id: str, default_config: DefaultTabiyaConfig
    ) -> PipelineDocument:
        existing = await self._repository.list_for_user(user_id)
        for pipeline in existing:
            if pipeline.is_default:
                return pipeline

        now = datetime.now(timezone.utc)
        # We activate the seeded default only when the user has no other
        # active pipeline; leaves an existing custom active pipeline in
        # place if one is somehow there before the default was seeded.
        should_activate = not any(pipeline.is_active for pipeline in existing)
        seed_doc = PipelineDocument(
            pipeline_id=str(uuid.uuid4()),
            user_id=user_id,
            name=DEFAULT_TABIYA_NAME,
            stages=build_default_tabiya_stages(
                nel_model_id=default_config.nel_model_id,
                taxonomy_model_id=default_config.taxonomy_model_id,
                top_k=default_config.top_k,
                min_similarity=default_config.min_similarity,
            ),
            is_active=should_activate,
            is_default=True,
            is_readonly=True,
            created_at=now,
            updated_at=now,
        )
        try:
            await self._repository.insert(seed_doc)
        except DuplicateKeyError:
            # A concurrent request seeded the default first. Re-fetch and return it.
            _logger.info(
                "ensure_default: concurrent insert detected for user=%s, re-fetching",
                user_id,
            )
            existing = await self._repository.list_for_user(user_id)
            for pipeline in existing:
                if pipeline.is_default:
                    return pipeline
            raise
        _logger.info(
            "Seeded Default Tabiya pipeline for user=%s (active=%s)",
            user_id,
            should_activate,
        )
        return seed_doc

    @staticmethod
    def _clone_name(source_name: str) -> str:
        suffix = " (copy)"
        if source_name.endswith(suffix):
            return source_name
        return f"{source_name}{suffix}"

"""Pipelines MongoDB repository.

Collection: `pipelines`.

Indexes:
  * `(user_id)` — non-unique, powers `list_for_user` cheaply.
  * `(user_id, is_active)` partial-unique where `is_active=True` — the DB
    enforces the "at most one active pipeline per user" invariant. Any
    would-be-racing code that tries to activate a second pipeline hits a
    `DuplicateKeyError`, which we catch and translate into a clean retry
    inside `set_active`.

The repository intentionally does NOT enforce validity of the pipeline
graph (source-first / sink-last / slot compatibility). That's the
validator's job in 11.5 — keeping it out of the persistence layer means
we can seed the Default Tabiya row without the validator being wired yet.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError

from ._types import PipelineDocument
from .errors import PipelineNotFoundError, ReadonlyPipelineError

_logger = logging.getLogger(__name__)

PIPELINES_COLLECTION = "pipelines"
USER_INDEX_NAME = "pipelines_user_idx"
ACTIVE_UNIQUE_INDEX_NAME = "pipelines_active_unique_idx"


class IPipelineRepository(ABC):
    """Protocol the service + tests depend on."""

    @abstractmethod
    async def ensure_indexes(self) -> None: ...

    @abstractmethod
    async def list_for_user(self, user_id: str) -> list[PipelineDocument]: ...

    @abstractmethod
    async def get(self, *, user_id: str, pipeline_id: str) -> Optional[PipelineDocument]: ...

    @abstractmethod
    async def insert(self, doc: PipelineDocument) -> None: ...

    @abstractmethod
    async def update(self, doc: PipelineDocument) -> PipelineDocument: ...

    @abstractmethod
    async def delete(self, *, user_id: str, pipeline_id: str) -> None: ...

    @abstractmethod
    async def set_active(self, *, user_id: str, pipeline_id: str) -> PipelineDocument: ...


class PipelineRepository(IPipelineRepository):
    def __init__(self, app_db: AsyncIOMotorDatabase) -> None:
        self._col = app_db[PIPELINES_COLLECTION]

    async def ensure_indexes(self) -> None:
        """Create indexes if missing. Idempotent — safe to call every startup."""

        await self._col.create_index(
            [("user_id", ASCENDING)], name=USER_INDEX_NAME
        )
        # Partial unique index: at most one active pipeline per user. Setting
        # `is_active=false` on another doc creates space for a new active
        # entry without violating the constraint. `partialFilterExpression`
        # is what makes it a "partial" index (non-active rows are simply
        # not indexed here).
        await self._col.create_index(
            [("user_id", ASCENDING), ("is_active", ASCENDING)],
            name=ACTIVE_UNIQUE_INDEX_NAME,
            unique=True,
            partialFilterExpression={"is_active": True},
        )

    async def list_for_user(self, user_id: str) -> list[PipelineDocument]:
        cursor = self._col.find(
            {"user_id": user_id},
            {"_id": 0},
        ).sort("created_at", ASCENDING)
        docs = await cursor.to_list(length=None)
        return [PipelineDocument.model_validate(doc) for doc in docs]

    async def get(
        self, *, user_id: str, pipeline_id: str
    ) -> Optional[PipelineDocument]:
        doc = await self._col.find_one(
            {"user_id": user_id, "pipeline_id": pipeline_id}, {"_id": 0}
        )
        if doc is None:
            return None
        return PipelineDocument.model_validate(doc)

    async def insert(self, doc: PipelineDocument) -> None:
        await self._col.insert_one(doc.model_dump())

    async def update(self, doc: PipelineDocument) -> PipelineDocument:
        """Update every mutable field on an existing pipeline.

        Refuses read-only rows. Uses `find_one_and_update` so the caller
        can see the freshly-persisted state (including the bumped
        `updated_at`).
        """

        existing = await self.get(user_id=doc.user_id, pipeline_id=doc.pipeline_id)
        if existing is None:
            raise PipelineNotFoundError(doc.pipeline_id, doc.user_id)
        if existing.is_readonly:
            raise ReadonlyPipelineError(doc.pipeline_id)

        payload = doc.model_dump()
        payload["updated_at"] = datetime.now(timezone.utc)

        # `is_readonly` and `is_default` are set at seed time; ignore any
        # attempt to flip them via update. `is_active` is only set via
        # set_active so we don't accidentally break the partial unique
        # index with a plain update.
        payload["is_readonly"] = existing.is_readonly
        payload["is_default"] = existing.is_default
        payload["is_active"] = existing.is_active
        payload["created_at"] = existing.created_at

        result = await self._col.find_one_and_update(
            {"user_id": doc.user_id, "pipeline_id": doc.pipeline_id},
            {"$set": payload},
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
        if result is None:
            raise PipelineNotFoundError(doc.pipeline_id, doc.user_id)
        return PipelineDocument.model_validate(result)

    async def delete(self, *, user_id: str, pipeline_id: str) -> None:
        existing = await self.get(user_id=user_id, pipeline_id=pipeline_id)
        if existing is None:
            raise PipelineNotFoundError(pipeline_id, user_id)
        if existing.is_readonly:
            raise ReadonlyPipelineError(pipeline_id)
        await self._col.delete_one(
            {"user_id": user_id, "pipeline_id": pipeline_id}
        )

    async def set_active(
        self, *, user_id: str, pipeline_id: str
    ) -> PipelineDocument:
        """Make `pipeline_id` the single active pipeline for the user.

        Two-step swap. The DB-level partial-unique index guarantees the
        "at most one active per user" invariant even if this method is
        raced against itself — the second caller's activate step hits
        `DuplicateKeyError` and we translate that into a retry.
        """

        target = await self.get(user_id=user_id, pipeline_id=pipeline_id)
        if target is None:
            raise PipelineNotFoundError(pipeline_id, user_id)

        # Step 1: deactivate every other active pipeline for this user.
        await self._col.update_many(
            {
                "user_id": user_id,
                "is_active": True,
                "pipeline_id": {"$ne": pipeline_id},
            },
            {"$set": {"is_active": False, "updated_at": datetime.now(timezone.utc)}},
        )

        # Step 2: activate the target. If the deactivate step somehow
        # didn't clear the previous active (crash between steps on an
        # earlier invocation), the partial-unique index would fire.
        # We retry once by re-running the deactivate + activate.
        try:
            result = await self._col.find_one_and_update(
                {"user_id": user_id, "pipeline_id": pipeline_id},
                {"$set": {"is_active": True, "updated_at": datetime.now(timezone.utc)}},
                projection={"_id": 0},
                return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError:
            _logger.warning(
                "set_active: DuplicateKeyError for user=%s pipeline=%s; retrying",
                user_id,
                pipeline_id,
            )
            await self._col.update_many(
                {"user_id": user_id, "is_active": True},
                {"$set": {"is_active": False}},
            )
            result = await self._col.find_one_and_update(
                {"user_id": user_id, "pipeline_id": pipeline_id},
                {"$set": {"is_active": True, "updated_at": datetime.now(timezone.utc)}},
                projection={"_id": 0},
                return_document=ReturnDocument.AFTER,
            )

        if result is None:
            raise PipelineNotFoundError(pipeline_id, user_id)
        return PipelineDocument.model_validate(result)

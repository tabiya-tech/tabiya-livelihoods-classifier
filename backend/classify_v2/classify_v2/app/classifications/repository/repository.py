"""Classifications MongoDB repository.

Collection: `classifications`.

Indexes:
  * `(user_id, created_at DESC)` — powers list_for_user and the usage
    aggregation efficiently. Covering index for the common query shape.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from ._types import ClassificationRecord

_logger = logging.getLogger(__name__)

CLASSIFICATIONS_COLLECTION = "classifications"
USER_INDEX_NAME = "classifications_user_created_idx"
CREATED_AT_INDEX_NAME = "classifications_created_at_idx"

_DEFAULT_PAGE_SIZE = 20
_MAX_PAGE_SIZE = 100


class IClassificationRepository(ABC):
    @abstractmethod
    async def ensure_indexes(self) -> None: ...

    @abstractmethod
    async def insert(self, record: ClassificationRecord) -> None: ...

    @abstractmethod
    async def list_for_user(
        self,
        user_id: str,
        *,
        limit: int = _DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
        entity_type: str | None = None,
    ) -> tuple[list[ClassificationRecord], str | None]: ...

    @abstractmethod
    async def daily_counts_for_user(
        self,
        user_id: str,
        *,
        days: int,
    ) -> list[dict]: ...


class ClassificationRepository(IClassificationRepository):
    def __init__(self, app_db: AsyncIOMotorDatabase) -> None:
        self._col = app_db[CLASSIFICATIONS_COLLECTION]

    async def ensure_indexes(self) -> None:
        """Create indexes if missing. Idempotent — safe to call every startup."""
        await self._col.create_index(
            [("user_id", ASCENDING), ("created_at", DESCENDING)],
            name=USER_INDEX_NAME,
        )

    async def insert(self, record: ClassificationRecord) -> None:
        await self._col.insert_one(record.model_dump())

    async def list_for_user(
        self,
        user_id: str,
        *,
        limit: int = _DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
        entity_type: str | None = None,
    ) -> tuple[list[ClassificationRecord], str | None]:
        """Return (records, next_cursor).

        Cursor is the `classification_id` of the last item returned.
        Fetch `limit + 1` to detect whether there is a next page without
        an extra COUNT query.

        `entity_type` filter is reserved for when the thin record gains a
        dominant-type field. Currently ignored.
        """
        clamped_limit = min(max(1, limit), _MAX_PAGE_SIZE)

        query: dict = {"user_id": user_id}
        if cursor:
            query["classification_id"] = {"$lt": cursor}

        raw_docs = (
            await self._col.find(query, {"_id": 0})
            .sort("created_at", DESCENDING)
            .limit(clamped_limit + 1)
            .to_list(length=clamped_limit + 1)
        )

        has_next = len(raw_docs) > clamped_limit
        page = raw_docs[:clamped_limit]
        records = [ClassificationRecord.model_validate(doc) for doc in page]
        next_cursor = records[-1].classification_id if has_next and records else None
        return records, next_cursor

    async def daily_counts_for_user(
        self,
        user_id: str,
        *,
        days: int,
    ) -> list[dict]:
        """Aggregate call counts per calendar day (UTC) for the last `days` days.

        Returns a list of `{"date": "YYYY-MM-DD", "count": N}` dicts sorted
        ascending by date. Days with zero calls are omitted — the frontend
        fills gaps when rendering the chart.
        """
        from datetime import datetime, timedelta, timezone

        since = datetime.now(timezone.utc) - timedelta(days=days)

        pipeline = [
            {"$match": {"user_id": user_id, "created_at": {"$gte": since}}},
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$created_at",
                            "timezone": "UTC",
                        }
                    },
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id": ASCENDING}},
            {"$project": {"_id": 0, "date": "$_id", "count": 1}},
        ]

        return await self._col.aggregate(pipeline).to_list(length=None)

"""Tests for /v2/classifications and /v2/usage routes.

Uses dependency overrides to inject a stub repository — the routes are a
thin translation layer, so tests verify the translation without Mongo.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from classify_v2.app.auth.firebase import get_firebase_uid
from classify_v2.app.classifications.repository import (
    ClassificationRecord,
    IClassificationRepository,
)
from classify_v2.app.classifications.routes.routes import (
    _get_classifications_repo,
    router as classifications_router,
)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _record(
    classification_id: str,
    user_id: str = "uid-1",
    pipeline_id: str = "pipeline-1",
    entity_count: int = 5,
    processing_time_ms: float = 100.0,
    created_at: datetime | None = None,
) -> ClassificationRecord:
    return ClassificationRecord(
        classification_id=classification_id,
        user_id=user_id,
        pipeline_id=pipeline_id,
        entity_count=entity_count,
        processing_time_ms=processing_time_ms,
        created_at=created_at or _now(),
    )


class _StubRepo(IClassificationRepository):
    def __init__(
        self,
        records: list[ClassificationRecord] | None = None,
        next_cursor: str | None = None,
        daily_counts: list[dict] | None = None,
    ) -> None:
        self._records = records or []
        self._next_cursor = next_cursor
        self._daily_counts = daily_counts or []

    async def ensure_indexes(self) -> None:
        pass

    async def insert(self, record: ClassificationRecord) -> None:
        self._records.append(record)

    async def list_for_user(
        self,
        user_id: str,
        *,
        limit: int = 20,
        cursor: str | None = None,
        entity_type: str | None = None,
    ) -> tuple[list[ClassificationRecord], str | None]:
        return self._records, self._next_cursor

    async def daily_counts_for_user(
        self,
        user_id: str,
        *,
        days: int,
    ) -> list[dict]:
        return self._daily_counts


def _make_app(stub_repo: IClassificationRepository) -> TestClient:
    test_app = FastAPI()
    test_app.include_router(classifications_router)
    test_app.dependency_overrides[get_firebase_uid] = lambda: "uid-1"
    test_app.dependency_overrides[_get_classifications_repo] = lambda: stub_repo
    return TestClient(test_app)


class TestListClassifications:
    def test_returns_empty_list_when_no_records(self):
        # GIVEN a repo with no records
        givenRepo = _StubRepo(records=[])
        client = _make_app(givenRepo)

        # WHEN GET /v2/classifications
        response = client.get("/v2/classifications")

        # THEN 200 with empty items
        assert response.status_code == 200
        body = response.json()
        assert body["items"] == []
        assert body["next_cursor"] is None

    def test_returns_records_as_summaries(self):
        # GIVEN a repo with one record
        givenRecord = _record("c1", pipeline_id="p1", entity_count=3)
        givenRepo = _StubRepo(records=[givenRecord])
        client = _make_app(givenRepo)

        # WHEN GET /v2/classifications
        response = client.get("/v2/classifications")

        # THEN 200 with the record's fields
        assert response.status_code == 200
        items = response.json()["items"]
        expectedCount = 1
        assert len(items) == expectedCount
        assert items[0]["classification_id"] == "c1"
        assert items[0]["pipeline_id"] == "p1"
        assert items[0]["entity_count"] == 3

    def test_forwards_next_cursor_when_more_pages_exist(self):
        # GIVEN a repo that signals there's a next page
        givenRecord = _record("c1")
        givenRepo = _StubRepo(records=[givenRecord], next_cursor="c1")
        client = _make_app(givenRepo)

        # WHEN GET /v2/classifications
        response = client.get("/v2/classifications")

        # THEN next_cursor is present in the response
        assert response.status_code == 200
        assert response.json()["next_cursor"] == "c1"

    def test_rejects_limit_above_100(self):
        # GIVEN a stub repo
        givenRepo = _StubRepo()
        client = _make_app(givenRepo)

        # WHEN GET /v2/classifications?limit=200
        response = client.get("/v2/classifications?limit=200")

        # THEN 422 validation error
        assert response.status_code == 422

    def test_rejects_limit_below_1(self):
        # GIVEN a stub repo
        givenRepo = _StubRepo()
        client = _make_app(givenRepo)

        # WHEN GET /v2/classifications?limit=0
        response = client.get("/v2/classifications?limit=0")

        # THEN 422 validation error
        assert response.status_code == 422


class TestGetUsage:
    def test_returns_empty_data_when_no_records(self):
        # GIVEN a repo with no daily counts
        givenRepo = _StubRepo(daily_counts=[])
        client = _make_app(givenRepo)

        # WHEN GET /v2/usage
        response = client.get("/v2/usage")

        # THEN 200 with empty data and default days=30
        assert response.status_code == 200
        body = response.json()
        assert body["days"] == 30
        assert body["data"] == []

    def test_returns_daily_counts_from_repo(self):
        # GIVEN a repo with two days of counts
        givenCounts = [
            {"date": "2026-07-01", "count": 3},
            {"date": "2026-07-02", "count": 7},
        ]
        givenRepo = _StubRepo(daily_counts=givenCounts)
        client = _make_app(givenRepo)

        # WHEN GET /v2/usage?days=7
        response = client.get("/v2/usage?days=7")

        # THEN 200 with the counts and the requested days reflected
        assert response.status_code == 200
        body = response.json()
        assert body["days"] == 7
        expectedData = [
            {"date": "2026-07-01", "count": 3},
            {"date": "2026-07-02", "count": 7},
        ]
        assert body["data"] == expectedData

    def test_rejects_days_above_365(self):
        # GIVEN a stub repo
        givenRepo = _StubRepo()
        client = _make_app(givenRepo)

        # WHEN GET /v2/usage?days=400
        response = client.get("/v2/usage?days=400")

        # THEN 422 validation error
        assert response.status_code == 422

    def test_rejects_days_below_1(self):
        # GIVEN a stub repo
        givenRepo = _StubRepo()
        client = _make_app(givenRepo)

        # WHEN GET /v2/usage?days=0
        response = client.get("/v2/usage?days=0")

        # THEN 422 validation error
        assert response.status_code == 422

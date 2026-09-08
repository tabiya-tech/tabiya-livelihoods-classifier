"""ClassificationRepository tests against an in-memory Mongo."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from classify_v2.app.classifications.repository import (
    CLASSIFICATIONS_COLLECTION,
    USER_INDEX_NAME,
    ClassificationRecord,
    ClassificationRepository,
)


@pytest.fixture
async def repo(in_memory_application_database):
    repository = ClassificationRepository(app_db=in_memory_application_database)
    await repository.ensure_indexes()
    return repository


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _record(
    *,
    classification_id: str,
    user_id: str,
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


class TestEnsureIndexes:
    async def test_creates_user_created_at_index(
        self, in_memory_application_database, repo
    ):
        # GIVEN a repository whose ensure_indexes has run
        givenDb = in_memory_application_database

        # WHEN we ask Mongo for the collection's indexes
        indexes = await givenDb[CLASSIFICATIONS_COLLECTION].index_information()

        # THEN the named index exists
        assert USER_INDEX_NAME in indexes


class TestInsert:
    async def test_insert_persists_record(self, repo, in_memory_application_database):
        # GIVEN a classification record
        givenRecord = _record(classification_id="c1", user_id="uid-1")

        # WHEN we insert it
        await repo.insert(givenRecord)

        # THEN one document is in the collection
        givenDb = in_memory_application_database
        count = await givenDb[CLASSIFICATIONS_COLLECTION].count_documents(
            {"user_id": "uid-1"}
        )
        expectedCount = 1
        assert count == expectedCount


class TestListForUser:
    async def test_returns_empty_when_no_records(self, repo):
        # GIVEN no records
        # WHEN list_for_user is called
        records, next_cursor = await repo.list_for_user("uid-1")

        # THEN empty list and no cursor
        assert records == []
        assert next_cursor is None

    async def test_returns_only_records_for_the_user(self, repo):
        # GIVEN records for two users
        await repo.insert(_record(classification_id="c-a", user_id="uid-1"))
        await repo.insert(_record(classification_id="c-b", user_id="uid-2"))

        # WHEN uid-1 lists
        records, _ = await repo.list_for_user("uid-1")

        # THEN only uid-1's record is returned
        expectedIds = ["c-a"]
        assert [record.classification_id for record in records] == expectedIds

    async def test_returns_records_sorted_newest_first(self, repo):
        # GIVEN three records with distinct created_at values
        now = _now()
        await repo.insert(_record(classification_id="oldest", user_id="uid-1", created_at=now))
        await repo.insert(_record(classification_id="middle", user_id="uid-1", created_at=now + timedelta(seconds=10)))
        await repo.insert(_record(classification_id="newest", user_id="uid-1", created_at=now + timedelta(seconds=20)))

        # WHEN we list
        records, _ = await repo.list_for_user("uid-1")

        # THEN they are newest-first
        expectedOrder = ["newest", "middle", "oldest"]
        assert [record.classification_id for record in records] == expectedOrder

    async def test_pagination_returns_next_cursor_when_more_exist(self, repo):
        # GIVEN 3 records
        now = _now()
        for index in range(3):
            await repo.insert(
                _record(
                    classification_id=f"c{index}",
                    user_id="uid-1",
                    created_at=now + timedelta(seconds=index),
                )
            )

        # WHEN we request limit=2
        records, next_cursor = await repo.list_for_user("uid-1", limit=2)

        # THEN 2 records are returned and a cursor is provided
        expectedCount = 2
        assert len(records) == expectedCount
        assert next_cursor is not None

    async def test_pagination_returns_no_cursor_on_last_page(self, repo):
        # GIVEN 2 records
        now = _now()
        for index in range(2):
            await repo.insert(
                _record(
                    classification_id=f"c{index}",
                    user_id="uid-1",
                    created_at=now + timedelta(seconds=index),
                )
            )

        # WHEN we request limit=5 (more than exist)
        records, next_cursor = await repo.list_for_user("uid-1", limit=5)

        # THEN all records returned and cursor is None
        expectedCount = 2
        assert len(records) == expectedCount
        assert next_cursor is None


class TestDailyCountsForUser:
    async def test_returns_empty_when_no_records(self, repo):
        # GIVEN no records
        # WHEN we aggregate
        result = await repo.daily_counts_for_user("uid-1", days=30)

        # THEN empty list
        assert result == []

    async def test_counts_calls_per_day(self, repo):
        # GIVEN 3 records on one day and 2 on another
        day_one = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
        day_two = datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)
        for index in range(3):
            await repo.insert(
                _record(
                    classification_id=f"d1-{index}",
                    user_id="uid-1",
                    created_at=day_one + timedelta(seconds=index),
                )
            )
        for index in range(2):
            await repo.insert(
                _record(
                    classification_id=f"d2-{index}",
                    user_id="uid-1",
                    created_at=day_two + timedelta(seconds=index),
                )
            )

        # WHEN we aggregate over a wide window that covers both days
        result = await repo.daily_counts_for_user("uid-1", days=365)

        # THEN we get one entry per day with the correct counts
        expectedDates = {"2026-07-01", "2026-07-02"}
        assert {entry["date"] for entry in result} == expectedDates
        countByDate = {entry["date"]: entry["count"] for entry in result}
        assert countByDate["2026-07-01"] == 3
        assert countByDate["2026-07-02"] == 2

    async def test_excludes_records_outside_the_window(self, repo):
        # GIVEN one old record (outside window) and one recent record
        now = datetime.now(timezone.utc)
        await repo.insert(
            _record(
                classification_id="old",
                user_id="uid-1",
                created_at=now - timedelta(days=60),
            )
        )
        await repo.insert(
            _record(
                classification_id="recent",
                user_id="uid-1",
                created_at=now - timedelta(days=1),
            )
        )

        # WHEN we aggregate over 30 days
        result = await repo.daily_counts_for_user("uid-1", days=30)

        # THEN only the recent record is counted
        total = sum(entry["count"] for entry in result)
        expectedTotal = 1
        assert total == expectedTotal

    async def test_does_not_include_other_users_records(self, repo):
        # GIVEN records for two users on the same day
        now = datetime.now(timezone.utc)
        await repo.insert(_record(classification_id="u1", user_id="uid-1", created_at=now))
        await repo.insert(_record(classification_id="u2", user_id="uid-2", created_at=now))

        # WHEN uid-1 fetches usage
        result = await repo.daily_counts_for_user("uid-1", days=30)

        # THEN only uid-1's record is counted
        total = sum(entry["count"] for entry in result)
        expectedTotal = 1
        assert total == expectedTotal

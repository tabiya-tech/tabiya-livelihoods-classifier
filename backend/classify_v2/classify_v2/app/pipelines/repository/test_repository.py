"""Pipelines repository tests against an in-memory Mongo."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pymongo.errors import DuplicateKeyError

from classify_v2.app.pipelines.repository import (
    ACTIVE_UNIQUE_INDEX_NAME,
    PIPELINES_COLLECTION,
    PipelineDocument,
    PipelineNotFoundError,
    PipelineRepository,
    ReadonlyPipelineError,
    StageDocument,
    USER_INDEX_NAME,
)


@pytest.fixture
async def repo(in_memory_application_database):
    repository = PipelineRepository(app_db=in_memory_application_database)
    await repository.ensure_indexes()
    return repository


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _pipeline(
    *,
    pipeline_id: str,
    user_id: str,
    name: str = "test pipeline",
    is_active: bool = False,
    is_default: bool = False,
    is_readonly: bool = False,
    stages: list[dict[str, Any]] | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> PipelineDocument:
    now = _now()
    return PipelineDocument(
        pipeline_id=pipeline_id,
        user_id=user_id,
        name=name,
        stages=[
            StageDocument(**stage)
            for stage in (
                stages
                if stages is not None
                else [
                    {"plugin_id": "tabiya.source.text.v1", "config": {}},
                    {"plugin_id": "tabiya.ner.v1", "config": {}},
                    {"plugin_id": "tabiya.nel.v1", "config": {}},
                    {"plugin_id": "tabiya.sink.results.v1", "config": {}},
                ]
            )
        ],
        is_active=is_active,
        is_default=is_default,
        is_readonly=is_readonly,
        created_at=created_at or now,
        updated_at=updated_at or now,
    )


class TestEnsureIndexes:
    async def test_creates_user_and_active_partial_unique_indexes(
        self, in_memory_application_database, repo
    ):
        # GIVEN a repository whose ensure_indexes has run
        givenDb = in_memory_application_database

        # WHEN we ask Mongo for the collection's indexes
        indexes = await givenDb[PIPELINES_COLLECTION].index_information()

        # THEN both named indexes exist
        expectedNames = {USER_INDEX_NAME, ACTIVE_UNIQUE_INDEX_NAME}
        assert expectedNames <= set(indexes.keys())

        # AND the active index is partial-unique on is_active=True
        activeIndex = indexes[ACTIVE_UNIQUE_INDEX_NAME]
        assert activeIndex.get("unique") is True
        assert activeIndex.get("partialFilterExpression") == {"is_active": True}


class TestListAndGet:
    async def test_list_for_user_returns_empty_when_no_pipelines(self, repo):
        # GIVEN no pipelines
        # WHEN list_for_user is called
        result = await repo.list_for_user("uid-1")

        # THEN empty list
        assert result == []

    async def test_insert_then_list_returns_the_pipeline(self, repo):
        # GIVEN one pipeline inserted for a user
        givenPipeline = _pipeline(pipeline_id="p1", user_id="uid-1")
        await repo.insert(givenPipeline)

        # WHEN we list for that user
        listed = await repo.list_for_user("uid-1")

        # THEN one entry with matching id
        expectedIds = ["p1"]
        assert [pipeline.pipeline_id for pipeline in listed] == expectedIds

    async def test_pipelines_are_isolated_per_user(self, repo):
        # GIVEN pipelines for two users
        await repo.insert(_pipeline(pipeline_id="p-a", user_id="uid-1"))
        await repo.insert(_pipeline(pipeline_id="p-b", user_id="uid-2"))

        # WHEN each user lists
        listed_1 = await repo.list_for_user("uid-1")
        listed_2 = await repo.list_for_user("uid-2")

        # THEN each sees only their own
        assert [pipeline.pipeline_id for pipeline in listed_1] == ["p-a"]
        assert [pipeline.pipeline_id for pipeline in listed_2] == ["p-b"]

    async def test_list_is_sorted_by_created_at_ascending(self, repo):
        # GIVEN three pipelines with distinct created_at values
        now = _now()
        await repo.insert(_pipeline(pipeline_id="second", user_id="uid-1", created_at=now + timedelta(seconds=10)))
        await repo.insert(_pipeline(pipeline_id="first", user_id="uid-1", created_at=now))
        await repo.insert(_pipeline(pipeline_id="third", user_id="uid-1", created_at=now + timedelta(seconds=20)))

        # WHEN we list
        listed = await repo.list_for_user("uid-1")

        # THEN they're ascending by created_at
        expectedOrder = ["first", "second", "third"]
        assert [pipeline.pipeline_id for pipeline in listed] == expectedOrder

    async def test_get_returns_none_when_missing(self, repo):
        # GIVEN no pipelines
        # WHEN we get a missing id
        result = await repo.get(user_id="uid-1", pipeline_id="nope")

        # THEN None
        assert result is None

    async def test_get_returns_the_pipeline_when_present(self, repo):
        # GIVEN an inserted pipeline
        givenPipeline = _pipeline(pipeline_id="p1", user_id="uid-1")
        await repo.insert(givenPipeline)

        # WHEN we fetch it
        fetched = await repo.get(user_id="uid-1", pipeline_id="p1")

        # THEN we get it back
        assert fetched is not None
        assert fetched.pipeline_id == "p1"

    async def test_get_never_returns_another_users_pipeline(self, repo):
        # GIVEN a pipeline owned by uid-1
        await repo.insert(_pipeline(pipeline_id="p1", user_id="uid-1"))

        # WHEN uid-2 tries to read it
        result = await repo.get(user_id="uid-2", pipeline_id="p1")

        # THEN None — cross-tenant isolation
        assert result is None


class TestUpdate:
    async def test_update_persists_new_name_and_stages(self, repo):
        # GIVEN a pipeline with the default stages
        givenPipeline = _pipeline(pipeline_id="p1", user_id="uid-1", name="original")
        await repo.insert(givenPipeline)

        # WHEN we update its name and stages
        updated = givenPipeline.model_copy(
            update={
                "name": "renamed",
                "stages": [
                    StageDocument(plugin_id="tabiya.source.text.v1"),
                    StageDocument(plugin_id="tabiya.sink.results.v1"),
                ],
            }
        )
        result = await repo.update(updated)

        # THEN the returned doc reflects the change AND has a fresh updated_at
        # (Mongo strips tzinfo on the roundtrip, so compare on naive UTC.)
        assert result.name == "renamed"
        expectedStageCount = 2
        assert len(result.stages) == expectedStageCount
        result_updated_naive = result.updated_at.replace(tzinfo=None)
        given_updated_naive = givenPipeline.updated_at.replace(tzinfo=None)
        assert result_updated_naive >= given_updated_naive

    async def test_update_missing_pipeline_raises_not_found(self, repo):
        # GIVEN nothing persisted
        givenPipeline = _pipeline(pipeline_id="ghost", user_id="uid-1")

        # WHEN we try to update it
        # THEN we get PipelineNotFoundError
        with pytest.raises(PipelineNotFoundError):
            await repo.update(givenPipeline)

    async def test_update_readonly_pipeline_raises(self, repo):
        # GIVEN a read-only pipeline (like the seeded Default)
        givenPipeline = _pipeline(
            pipeline_id="default", user_id="uid-1", is_readonly=True
        )
        await repo.insert(givenPipeline)

        # WHEN we try to update it
        # THEN ReadonlyPipelineError
        with pytest.raises(ReadonlyPipelineError):
            await repo.update(givenPipeline.model_copy(update={"name": "hijacked"}))

    async def test_update_preserves_readonly_default_and_active_flags(self, repo):
        # GIVEN a non-readonly, non-default, non-active pipeline
        givenPipeline = _pipeline(
            pipeline_id="p1",
            user_id="uid-1",
            is_readonly=False,
            is_default=False,
            is_active=False,
        )
        await repo.insert(givenPipeline)

        # WHEN we try to update with the flags flipped
        malicious = givenPipeline.model_copy(
            update={"is_readonly": True, "is_default": True, "is_active": True}
        )
        result = await repo.update(malicious)

        # THEN the flags are preserved from the persisted state, not the input
        assert result.is_readonly is False
        assert result.is_default is False
        assert result.is_active is False


class TestDelete:
    async def test_delete_removes_the_pipeline(self, repo):
        # GIVEN one pipeline
        await repo.insert(_pipeline(pipeline_id="p1", user_id="uid-1"))

        # WHEN we delete it
        await repo.delete(user_id="uid-1", pipeline_id="p1")

        # THEN it's gone
        assert await repo.list_for_user("uid-1") == []

    async def test_delete_missing_raises_not_found(self, repo):
        # GIVEN nothing persisted
        # WHEN we delete
        # THEN PipelineNotFoundError
        with pytest.raises(PipelineNotFoundError):
            await repo.delete(user_id="uid-1", pipeline_id="ghost")

    async def test_delete_readonly_raises(self, repo):
        # GIVEN a readonly pipeline
        await repo.insert(
            _pipeline(pipeline_id="default", user_id="uid-1", is_readonly=True)
        )

        # WHEN we try to delete
        # THEN ReadonlyPipelineError
        with pytest.raises(ReadonlyPipelineError):
            await repo.delete(user_id="uid-1", pipeline_id="default")


class TestSetActive:
    async def test_set_active_flips_the_designated_pipeline(self, repo):
        # GIVEN two inactive pipelines
        await repo.insert(_pipeline(pipeline_id="p1", user_id="uid-1"))
        await repo.insert(_pipeline(pipeline_id="p2", user_id="uid-1"))

        # WHEN we activate p1
        result = await repo.set_active(user_id="uid-1", pipeline_id="p1")

        # THEN p1 is now active, p2 remains inactive
        assert result.is_active is True
        assert result.pipeline_id == "p1"
        listed = await repo.list_for_user("uid-1")
        assert {pipeline.pipeline_id: pipeline.is_active for pipeline in listed} == {
            "p1": True,
            "p2": False,
        }

    async def test_set_active_deactivates_the_previous_active(self, repo):
        # GIVEN p1 active, p2 inactive
        await repo.insert(_pipeline(pipeline_id="p1", user_id="uid-1", is_active=True))
        await repo.insert(_pipeline(pipeline_id="p2", user_id="uid-1"))

        # WHEN we activate p2
        await repo.set_active(user_id="uid-1", pipeline_id="p2")

        # THEN p1 is no longer active
        listed = {p.pipeline_id: p.is_active for p in await repo.list_for_user("uid-1")}
        assert listed == {"p1": False, "p2": True}

    async def test_set_active_missing_raises_not_found(self, repo):
        # GIVEN one existing pipeline
        await repo.insert(_pipeline(pipeline_id="p1", user_id="uid-1"))

        # WHEN we try to activate a non-existent one
        # THEN PipelineNotFoundError
        with pytest.raises(PipelineNotFoundError):
            await repo.set_active(user_id="uid-1", pipeline_id="ghost")

    async def test_set_active_leaves_other_users_untouched(self, repo):
        # GIVEN two users each with an active pipeline
        await repo.insert(
            _pipeline(pipeline_id="p-a", user_id="uid-1", is_active=True)
        )
        await repo.insert(
            _pipeline(pipeline_id="p-b", user_id="uid-2", is_active=True)
        )
        # AND an extra inactive pipeline for uid-1
        await repo.insert(_pipeline(pipeline_id="p-a2", user_id="uid-1"))

        # WHEN uid-1 activates their other pipeline
        await repo.set_active(user_id="uid-1", pipeline_id="p-a2")

        # THEN uid-2's active pipeline stays active
        listed_2 = {p.pipeline_id: p.is_active for p in await repo.list_for_user("uid-2")}
        assert listed_2 == {"p-b": True}


class TestActivePartialUniqueIndex:
    async def test_direct_insert_of_second_active_row_hits_duplicate_key(
        self, in_memory_application_database, repo
    ):
        # GIVEN one active pipeline
        await repo.insert(_pipeline(pipeline_id="p1", user_id="uid-1", is_active=True))

        # WHEN we try to sneak a second active row past the repository layer
        col = in_memory_application_database[PIPELINES_COLLECTION]
        second = _pipeline(pipeline_id="p2", user_id="uid-1", is_active=True).model_dump()

        # THEN Mongo rejects with DuplicateKeyError — the partial-unique
        # index is the safety net for concurrent set_active calls.
        with pytest.raises(DuplicateKeyError):
            await col.insert_one(second)

    async def test_two_users_can_each_have_their_own_active_row(self, repo):
        # GIVEN two users each activating their own pipeline
        await repo.insert(
            _pipeline(pipeline_id="p-a", user_id="uid-1", is_active=True)
        )
        await repo.insert(
            _pipeline(pipeline_id="p-b", user_id="uid-2", is_active=True)
        )

        # WHEN we list each
        listed_1 = await repo.list_for_user("uid-1")
        listed_2 = await repo.list_for_user("uid-2")

        # THEN both users have an active pipeline — the partial-unique
        # index only constrains within a user_id, not globally
        assert [pipeline.is_active for pipeline in listed_1] == [True]
        assert [pipeline.is_active for pipeline in listed_2] == [True]

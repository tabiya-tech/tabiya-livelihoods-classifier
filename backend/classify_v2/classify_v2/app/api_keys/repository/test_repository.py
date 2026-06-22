"""Repository tests against an in-memory Mongo (same idiom as nel_v2)."""

import pytest

from classify_v2.app.api_keys.repository.repository import ApiKeysRepository
from classify_v2.app.api_keys.service.types import ApiKeyMetadata


@pytest.fixture
async def repo(in_memory_application_database):
    return ApiKeysRepository(app_db=in_memory_application_database)


def _meta(*, user_id: str, key_id: str, label: str = "test-key") -> ApiKeyMetadata:
    return ApiKeyMetadata(
        key_id=key_id,
        user_id=user_id,
        label=label,
        created_at=1700000000.0,
        last_used_at=None,
        revoked=False,
    )


class TestApiKeysRepository:
    async def test_list_active_returns_empty_when_no_keys(self, repo):
        # GIVEN no keys for the user
        # WHEN list_active is called
        result = await repo.list_active("uid-1")

        # THEN an empty list is returned
        assert result == []

    async def test_insert_then_list_returns_metadata_without_hash(self, repo):
        # GIVEN an inserted key
        await repo.insert(
            meta=_meta(user_id="uid-1", key_id="k1"),
            key_hash="sha256-deadbeef",
            gcp_key_name="projects/p/locations/global/keys/k1",
        )

        # WHEN list_active is called
        listed = await repo.list_active("uid-1")

        # THEN the metadata is returned and contains no secret fields
        assert len(listed) == 1
        assert listed[0].key_id == "k1"
        assert listed[0].user_id == "uid-1"
        # The model itself has no hash/gcp fields, so listing is structurally safe.

    async def test_list_excludes_revoked_keys(self, repo):
        # GIVEN one active and one revoked key for the same user
        await repo.insert(
            meta=_meta(user_id="uid-1", key_id="active"),
            key_hash="h1",
            gcp_key_name="g1",
        )
        await repo.insert(
            meta=_meta(user_id="uid-1", key_id="dead"),
            key_hash="h2",
            gcp_key_name="g2",
        )
        await repo.mark_revoked(user_id="uid-1", key_id="dead")

        # WHEN list_active is called
        listed = await repo.list_active("uid-1")

        # THEN only the active key surfaces
        assert [k.key_id for k in listed] == ["active"]

    async def test_keys_are_isolated_per_user(self, repo):
        # GIVEN keys for two users
        await repo.insert(meta=_meta(user_id="uid-1", key_id="a"), key_hash="x", gcp_key_name="g")
        await repo.insert(meta=_meta(user_id="uid-2", key_id="b"), key_hash="y", gcp_key_name="g")

        # WHEN each user lists their keys
        # THEN they only see their own
        assert [k.key_id for k in await repo.list_active("uid-1")] == ["a"]
        assert [k.key_id for k in await repo.list_active("uid-2")] == ["b"]

    async def test_count_active_matches_list(self, repo):
        # GIVEN three active keys for one user
        for i in range(3):
            await repo.insert(
                meta=_meta(user_id="uid-1", key_id=f"k{i}"),
                key_hash=f"h{i}",
                gcp_key_name=f"g{i}",
            )

        # WHEN count_active is called
        # THEN it matches the listed count
        assert await repo.count_active("uid-1") == 3

    async def test_find_gcp_key_name_returns_persisted_value(self, repo):
        # GIVEN a key with a known GCP resource name
        await repo.insert(
            meta=_meta(user_id="uid-1", key_id="k1"),
            key_hash="h",
            gcp_key_name="projects/test/locations/global/keys/k1",
        )

        # WHEN find_gcp_key_name is queried for the right user
        result = await repo.find_gcp_key_name(user_id="uid-1", key_id="k1")

        # THEN the stored value comes back
        assert result == "projects/test/locations/global/keys/k1"

    async def test_find_gcp_key_name_returns_none_for_other_user(self, repo):
        # GIVEN a key belonging to user A
        await repo.insert(
            meta=_meta(user_id="uid-A", key_id="k1"),
            key_hash="h",
            gcp_key_name="g",
        )

        # WHEN user B tries to look it up
        result = await repo.find_gcp_key_name(user_id="uid-B", key_id="k1")

        # THEN None comes back — no cross-tenant access
        assert result is None

    async def test_mark_revoked_returns_false_when_already_revoked(self, repo):
        # GIVEN a key that's already revoked
        await repo.insert(
            meta=_meta(user_id="uid-1", key_id="k1"),
            key_hash="h",
            gcp_key_name="g",
        )
        await repo.mark_revoked(user_id="uid-1", key_id="k1")

        # WHEN mark_revoked is called again
        result = await repo.mark_revoked(user_id="uid-1", key_id="k1")

        # THEN it reports no change
        assert result is False

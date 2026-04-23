"""Tests for UserConfigRepository."""

import pytest

from nel.app.user_config.repository.repository import UserConfigRepository
from nel.app.user_config.service.types import UserConfig


@pytest.fixture
async def repo(in_memory_application_database):
    return UserConfigRepository(app_db=in_memory_application_database)


class TestUserConfigRepository:
    async def test_get_returns_none_when_not_exists(self, repo):
        # GIVEN no config exists for a user
        # WHEN get is called
        result = await repo.get("user-1")

        # THEN None is returned
        assert result is None

    async def test_upsert_then_get_returns_config(self, repo):
        # GIVEN a config is upserted
        config = UserConfig(user_id="user-1", taxonomy_model_id="tax-1", nel_model_id="nel-1")
        await repo.upsert(config)

        # WHEN get is called
        result = await repo.get("user-1")

        # THEN the config is returned with correct fields
        assert result is not None
        assert result.user_id == "user-1"
        assert result.taxonomy_model_id == "tax-1"
        assert result.nel_model_id == "nel-1"

    async def test_upsert_updates_existing_config(self, repo):
        # GIVEN an existing config
        await repo.upsert(UserConfig(user_id="user-1", taxonomy_model_id="tax-1", nel_model_id="nel-1"))

        # WHEN upserted again with new values
        await repo.upsert(UserConfig(user_id="user-1", taxonomy_model_id="tax-2", nel_model_id="nel-2"))

        # THEN the updated values are returned
        result = await repo.get("user-1")
        assert result.taxonomy_model_id == "tax-2"
        assert result.nel_model_id == "nel-2"

    async def test_configs_are_isolated_per_user(self, repo):
        # GIVEN configs for two different users
        await repo.upsert(UserConfig(user_id="user-1", taxonomy_model_id="tax-1", nel_model_id="nel-1"))
        await repo.upsert(UserConfig(user_id="user-2", taxonomy_model_id="tax-2", nel_model_id="nel-2"))

        # WHEN each user's config is fetched
        result_1 = await repo.get("user-1")
        result_2 = await repo.get("user-2")

        # THEN each returns their own config
        assert result_1.taxonomy_model_id == "tax-1"
        assert result_2.taxonomy_model_id == "tax-2"

"""Tests for UserConfigService."""

from unittest.mock import AsyncMock

import pytest

from nel.app.user_config.repository.repository import IUserConfigRepository
from nel.app.user_config.service.service import UserConfigService
from nel.app.user_config.service.types import UserConfig


class FakeRepo(IUserConfigRepository):
    def __init__(self, existing: UserConfig | None = None):
        self._store: dict[str, UserConfig] = {}
        if existing:
            self._store[existing.user_id] = existing

    async def get(self, user_id: str) -> UserConfig | None:
        return self._store.get(user_id)

    async def upsert(self, config: UserConfig) -> None:
        self._store[config.user_id] = config


class TestUserConfigServiceGet:
    async def test_returns_config_when_exists(self):
        # GIVEN a config exists for the user
        existing = UserConfig(user_id="u1", taxonomy_model_id="tax-1", nel_model_id="nel-1")
        svc = UserConfigService(FakeRepo(existing))

        # WHEN get is called
        result = await svc.get("u1")

        # THEN the config is returned
        assert result.taxonomy_model_id == "tax-1"
        assert result.nel_model_id == "nel-1"

    async def test_returns_empty_config_when_not_exists(self):
        # GIVEN no config exists for the user
        svc = UserConfigService(FakeRepo())

        # WHEN get is called
        result = await svc.get("u1")

        # THEN an empty config is returned (not an error)
        assert result.user_id == "u1"
        assert result.taxonomy_model_id == ""
        assert result.nel_model_id == ""


class TestUserConfigServiceUpsert:
    async def test_upsert_stores_and_returns_config(self):
        # GIVEN no existing config
        repo = FakeRepo()
        svc = UserConfigService(repo)
        config = UserConfig(user_id="u1", taxonomy_model_id="tax-1", nel_model_id="nel-1")

        # WHEN upsert is called
        result = await svc.upsert(config)

        # THEN the config is returned and persisted
        assert result.taxonomy_model_id == "tax-1"
        assert repo._store["u1"].nel_model_id == "nel-1"

    async def test_upsert_overwrites_existing_config(self):
        # GIVEN an existing config
        existing = UserConfig(user_id="u1", taxonomy_model_id="tax-1", nel_model_id="nel-1")
        repo = FakeRepo(existing)
        svc = UserConfigService(repo)

        # WHEN upsert is called with new values
        updated = UserConfig(user_id="u1", taxonomy_model_id="tax-2", nel_model_id="nel-2")
        result = await svc.upsert(updated)

        # THEN the new values are returned and persisted
        assert result.taxonomy_model_id == "tax-2"
        assert repo._store["u1"].taxonomy_model_id == "tax-2"

"""Tests for user config routes."""

import pytest
from unittest.mock import patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from nel.app.user_config.routes.routes import router, _get_service
from nel.app.user_config.routes.auth import get_firebase_uid
from nel.app.user_config.service.service import IUserConfigService
from nel.app.user_config.service.types import UserConfig


class FakeUserConfigService(IUserConfigService):
    def __init__(self, existing: UserConfig | None = None):
        self._store: dict[str, UserConfig] = {}
        if existing:
            self._store[existing.user_id] = existing

    async def get(self, user_id: str) -> UserConfig:
        return self._store.get(user_id) or UserConfig(user_id=user_id, taxonomy_model_id="", nel_model_id="")

    async def upsert(self, config: UserConfig) -> UserConfig:
        self._store[config.user_id] = config
        return config


def _make_app(svc: IUserConfigService, uid: str = "user-1") -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[_get_service] = lambda: svc
    test_app.dependency_overrides[get_firebase_uid] = lambda: uid
    return test_app


class TestGetUserConfig:
    async def test_returns_config_when_exists(self):
        # GIVEN a config exists for the user
        existing = UserConfig(user_id="user-1", taxonomy_model_id="tax-1", nel_model_id="nel-1")
        svc = FakeUserConfigService(existing)

        # WHEN GET /v2/nel/user/config is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.get("/v2/nel/user/config")

        # THEN 200 is returned with the correct config
        assert resp.status_code == 200
        data = resp.json()
        assert data["taxonomy_model_id"] == "tax-1"
        assert data["nel_model_id"] == "nel-1"

    async def test_returns_empty_config_when_not_set(self):
        # GIVEN no config exists for the user
        svc = FakeUserConfigService()

        # WHEN GET /v2/nel/user/config is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.get("/v2/nel/user/config")

        # THEN 200 is returned with empty fields
        assert resp.status_code == 200
        data = resp.json()
        assert data["taxonomy_model_id"] == ""
        assert data["nel_model_id"] == ""

    async def test_api_key_auth_uses_shared_uid_in_production(self):
        # GIVEN the service is running in production mode (not local)
        test_app = FastAPI()
        test_app.include_router(router)
        test_app.dependency_overrides[_get_service] = lambda: FakeUserConfigService()

        # WHEN GET /v2/nel/user/config is called without the gateway user-info header
        # (i.e. authenticated via API key — the gateway has already verified the key
        # but does not set x-apigateway-api-userinfo for API key auth)
        with patch("nel.app.user_config.routes.auth.TARGET_ENVIRONMENT_TYPE", "production"):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                resp = await client.get("/v2/nel/user/config")

        # THEN 200 is returned using the shared api-key-user uid
        assert resp.status_code == 200


class TestUpdateUserConfig:
    async def test_stores_and_returns_updated_config(self):
        # GIVEN no existing config
        svc = FakeUserConfigService()

        # WHEN PUT /v2/nel/user/config is called with new values
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.put(
                "/v2/nel/user/config",
                json={"taxonomy_model_id": "tax-1", "nel_model_id": "nel-1"},
            )

        # THEN 200 is returned with the new config
        assert resp.status_code == 200
        data = resp.json()
        assert data["taxonomy_model_id"] == "tax-1"
        assert data["nel_model_id"] == "nel-1"

    async def test_overwrites_existing_config(self):
        # GIVEN an existing config
        existing = UserConfig(user_id="user-1", taxonomy_model_id="tax-old", nel_model_id="nel-old")
        svc = FakeUserConfigService(existing)

        # WHEN PUT /v2/nel/user/config is called with updated values
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.put(
                "/v2/nel/user/config",
                json={"taxonomy_model_id": "tax-new", "nel_model_id": "nel-new"},
            )

        # THEN the new values are returned
        assert resp.status_code == 200
        assert resp.json()["taxonomy_model_id"] == "tax-new"

    async def test_config_is_scoped_to_authenticated_user(self):
        # GIVEN two users with different uids
        svc = FakeUserConfigService()
        app_user1 = _make_app(svc, uid="user-1")
        app_user2 = _make_app(svc, uid="user-2")

        # WHEN each user sets their own config
        async with AsyncClient(transport=ASGITransport(app=app_user1), base_url="http://test") as client:
            await client.put("/v2/nel/user/config", json={"taxonomy_model_id": "tax-1", "nel_model_id": "nel-1"})
        async with AsyncClient(transport=ASGITransport(app=app_user2), base_url="http://test") as client:
            await client.put("/v2/nel/user/config", json={"taxonomy_model_id": "tax-2", "nel_model_id": "nel-2"})

        # THEN each user's GET returns their own config
        async with AsyncClient(transport=ASGITransport(app=app_user1), base_url="http://test") as client:
            resp = await client.get("/v2/nel/user/config")
        assert resp.json()["taxonomy_model_id"] == "tax-1"

        async with AsyncClient(transport=ASGITransport(app=app_user2), base_url="http://test") as client:
            resp = await client.get("/v2/nel/user/config")
        assert resp.json()["taxonomy_model_id"] == "tax-2"

    async def test_api_key_auth_uses_shared_uid_in_production(self):
        # GIVEN the service is running in production mode (not local)
        test_app = FastAPI()
        test_app.include_router(router)
        test_app.dependency_overrides[_get_service] = lambda: FakeUserConfigService()

        # WHEN PUT /v2/nel/user/config is called without the gateway user-info header
        # (i.e. authenticated via API key)
        with patch("nel.app.user_config.routes.auth.TARGET_ENVIRONMENT_TYPE", "production"):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                resp = await client.put(
                    "/v2/nel/user/config",
                    json={"taxonomy_model_id": "tax-1", "nel_model_id": "nel-1"},
                )

        # THEN 200 is returned using the shared api-key-user uid
        assert resp.status_code == 200

"""Tests for NEL models routes."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from nel.app.nel_models.repository.repository import INELModelRepository, NELModelInfo
from nel.app.nel_models.routes.routes import router, _get_service
from nel.app.nel_models.service.service import NELModelService


class FakeRepo(INELModelRepository):
    def __init__(self, models=None):
        self._models = {m.model_id: m for m in (models or [])}

    async def get_all(self):
        return list(self._models.values())

    async def get(self, model_id):
        return self._models.get(model_id)

    async def upsert(self, model):
        self._models[model.model_id] = model


def _make_app(models=None):
    app = FastAPI()
    app.include_router(router)
    fake_svc = NELModelService(FakeRepo(models or []))
    app.dependency_overrides[_get_service] = lambda: fake_svc
    return app


@pytest.fixture
def two_models():
    return [
        NELModelInfo(model_id="all-MiniLM-L6-v2", dimensions=384),
        NELModelInfo(model_id="mpnet-base-v2", dimensions=768),
    ]


class TestListNELModels:
    async def test_returns_all_models(self, two_models):
        # GIVEN two models exist
        # WHEN GET /v2/nel/models is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(two_models)), base_url="http://test") as client:
            resp = await client.get("/v2/nel/models")

        # THEN both are returned with status 200
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_returns_empty_list_when_no_models(self):
        # GIVEN no models exist
        # WHEN GET /v2/nel/models is called
        async with AsyncClient(transport=ASGITransport(app=_make_app()), base_url="http://test") as client:
            resp = await client.get("/v2/nel/models")

        # THEN an empty list is returned with status 200
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_response_contains_correct_fields(self, two_models):
        # GIVEN a model with known dimensions
        # WHEN GET /v2/nel/models is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(two_models)), base_url="http://test") as client:
            resp = await client.get("/v2/nel/models")

        # THEN each item has the expected fields
        models = resp.json()
        model_ids = {m["model_id"] for m in models}
        assert "all-MiniLM-L6-v2" in model_ids
        matching = next(m for m in models if m["model_id"] == "all-MiniLM-L6-v2")
        assert matching["dimensions"] == 384


class TestGetNELModel:
    async def test_returns_model_when_exists(self, two_models):
        # GIVEN a model with id "all-MiniLM-L6-v2" exists
        # WHEN GET /v2/nel/models/all-MiniLM-L6-v2 is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(two_models)), base_url="http://test") as client:
            resp = await client.get("/v2/nel/models/all-MiniLM-L6-v2")

        # THEN 200 is returned with correct dimensions
        assert resp.status_code == 200
        assert resp.json()["dimensions"] == 384

    async def test_returns_404_when_model_not_found(self):
        # GIVEN no models exist
        # WHEN GET /v2/nel/models/nonexistent is called
        async with AsyncClient(transport=ASGITransport(app=_make_app()), base_url="http://test") as client:
            resp = await client.get("/v2/nel/models/nonexistent")

        # THEN 404 is returned
        assert resp.status_code == 404

    async def test_returns_404_for_wrong_id_when_others_exist(self, two_models):
        # GIVEN two models exist but neither has id "unknown"
        # WHEN GET /v2/nel/models/unknown is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(two_models)), base_url="http://test") as client:
            resp = await client.get("/v2/nel/models/unknown")

        # THEN 404 is returned
        assert resp.status_code == 404

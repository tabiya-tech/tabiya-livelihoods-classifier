"""Tests for taxonomy models routes."""

import pytest
import httpx
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from nel.app.taxonomy_models.routes.routes import router, _get_service
from nel.app.taxonomy_models.service.errors import TaxonomyAPIError
from nel.app.taxonomy_models.service.service import ITaxonomyModelService
from nel.app.taxonomy_models.service.types import TaxonomyModelInfo


class FakeTaxonomyModelService(ITaxonomyModelService):
    def __init__(self, models: list[TaxonomyModelInfo] | None = None, raises: Exception | None = None):
        self._models = models or []
        self._raises = raises

    async def get_all(self) -> list[TaxonomyModelInfo]:
        if self._raises:
            raise self._raises
        return self._models


def _make_app(svc: ITaxonomyModelService) -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[_get_service] = lambda: svc
    return test_app


def _model(model_id="m1", name="ESCO 1.1.1") -> TaxonomyModelInfo:
    return TaxonomyModelInfo(id=model_id, name=name, version="v1.0.0", description="", released=True)


class TestListTaxonomyModels:
    async def test_returns_all_models(self):
        # GIVEN the taxonomy API returns two models
        svc = FakeTaxonomyModelService(models=[_model("m1", "Model 1"), _model("m2", "Model 2")])

        # WHEN GET /v2/nel/taxonomy-models is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.get("/v2/nel/taxonomy-models")

        # THEN both models are returned with status 200
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_returns_empty_list_when_no_models(self):
        # GIVEN the taxonomy API returns no models
        svc = FakeTaxonomyModelService(models=[])

        # WHEN GET /v2/nel/taxonomy-models is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.get("/v2/nel/taxonomy-models")

        # THEN an empty list is returned with status 200
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_response_contains_correct_fields(self):
        # GIVEN a model with known fields
        svc = FakeTaxonomyModelService(models=[
            TaxonomyModelInfo(id="abc123", name="ESCO 1.1.1", version="v1.0.0", description="A model", released=True)
        ])

        # WHEN GET /v2/nel/taxonomy-models is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.get("/v2/nel/taxonomy-models")

        # THEN the response contains the expected fields
        model = resp.json()[0]
        assert model["id"] == "abc123"
        assert model["name"] == "ESCO 1.1.1"
        assert model["version"] == "v1.0.0"
        assert model["released"] is True

    async def test_returns_502_when_taxonomy_api_fails(self):
        # GIVEN the taxonomy API is unavailable
        svc = FakeTaxonomyModelService(raises=TaxonomyAPIError("Taxonomy API unreachable"))

        # WHEN GET /v2/nel/taxonomy-models is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.get("/v2/nel/taxonomy-models")

        # THEN 502 is returned
        assert resp.status_code == 502

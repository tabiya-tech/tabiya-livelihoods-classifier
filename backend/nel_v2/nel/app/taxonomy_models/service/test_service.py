"""Tests for TaxonomyModelService."""

import pytest
import respx
import httpx

from nel.app.taxonomy_models.service.errors import TaxonomyAPIError
from nel.app.taxonomy_models.service.service import TaxonomyModelService


def _model_payload(**overrides) -> dict:
    defaults = {
        "id": "abc123",
        "name": "ESCO 1.1.1",
        "version": "v1.0.0",
        "description": "A test model",
        "released": True,
    }
    defaults.update(overrides)
    return defaults


class TestTaxonomyModelServiceGetAll:
    @respx.mock
    async def test_returns_models_from_api(self):
        # GIVEN the taxonomy API returns two models
        respx.get("https://taxonomy.example.com/api/app/models").mock(
            return_value=httpx.Response(200, json=[
                _model_payload(id="m1", name="Model 1"),
                _model_payload(id="m2", name="Model 2"),
            ])
        )
        svc = TaxonomyModelService(base_url="https://taxonomy.example.com")

        # WHEN get_all is called
        result = await svc.get_all()

        # THEN both models are returned with correct fields
        assert len(result) == 2
        assert result[0].id == "m1"
        assert result[0].name == "Model 1"
        assert result[1].id == "m2"

    @respx.mock
    async def test_returns_empty_list_when_api_returns_no_models(self):
        # GIVEN the taxonomy API returns an empty list
        respx.get("https://taxonomy.example.com/api/app/models").mock(
            return_value=httpx.Response(200, json=[])
        )
        svc = TaxonomyModelService(base_url="https://taxonomy.example.com")

        # WHEN get_all is called
        result = await svc.get_all()

        # THEN an empty list is returned
        assert result == []

    @respx.mock
    async def test_sends_api_key_header(self):
        # GIVEN an API key is configured
        route = respx.get("https://taxonomy.example.com/api/app/models").mock(
            return_value=httpx.Response(200, json=[])
        )
        svc = TaxonomyModelService(base_url="https://taxonomy.example.com", api_key="secret")

        # WHEN get_all is called
        await svc.get_all()

        # THEN the request includes the X-API-Key header
        assert route.calls[0].request.headers["x-api-key"] == "secret"

    @respx.mock
    async def test_raises_taxonomy_api_error_on_http_error(self):
        # GIVEN the taxonomy API returns a 500
        respx.get("https://taxonomy.example.com/api/app/models").mock(
            return_value=httpx.Response(500)
        )
        svc = TaxonomyModelService(base_url="https://taxonomy.example.com")

        # WHEN get_all is called
        with pytest.raises(TaxonomyAPIError):
            await svc.get_all()

    @respx.mock
    async def test_raises_taxonomy_api_error_on_network_error(self):
        # GIVEN the taxonomy API is unreachable
        respx.get("https://taxonomy.example.com/api/app/models").mock(
            side_effect=httpx.TransportError("connection refused")
        )
        svc = TaxonomyModelService(base_url="https://taxonomy.example.com")

        # WHEN get_all is called
        with pytest.raises(TaxonomyAPIError):
            await svc.get_all()

    @respx.mock
    async def test_only_released_flag_is_mapped_correctly(self):
        # GIVEN one released and one unreleased model
        respx.get("https://taxonomy.example.com/api/app/models").mock(
            return_value=httpx.Response(200, json=[
                _model_payload(id="m1", released=True),
                _model_payload(id="m2", released=False),
            ])
        )
        svc = TaxonomyModelService(base_url="https://taxonomy.example.com")

        # WHEN get_all is called
        result = await svc.get_all()

        # THEN released flags are correctly mapped
        assert result[0].released is True
        assert result[1].released is False

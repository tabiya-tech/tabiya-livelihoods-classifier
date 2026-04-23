"""Tests for entity linking routes."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from unittest.mock import patch

from nel.app.linking.routes.routes import router, _get_service
from nel.app.linking.service.errors import EmbeddingsCacheNotReadyError
from nel.app.linking.service.service import INELService
from nel.app.linking.service.types import (
    EntityType,
    LinkedEntity,
    NELMetadata,
    NELOptions,
    NELResponse,
    OccupationEntity,
    OccupationMatch,
)


class FakeNELService(INELService):
    def __init__(self, response: NELResponse | None = None, raises: Exception | None = None):
        self._response = response
        self._raises = raises
        self.last_call: dict | None = None

    async def link(self, entities, taxonomy_model_id, nel_model_id, options):
        self.last_call = dict(
            entities=entities,
            taxonomy_model_id=taxonomy_model_id,
            nel_model_id=nel_model_id,
            options=options,
        )
        if self._raises:
            raise self._raises
        return self._response


def _make_app(svc: INELService) -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[_get_service] = lambda: svc
    return test_app


def _make_response(label="Head Chef") -> NELResponse:
    return NELResponse(
        linked_entities=[
            LinkedEntity(
                input_text=label,
                entity_type=EntityType.occupation,
                matches=[
                    OccupationMatch(
                        similarity_score=0.92,
                        entity=OccupationEntity(
                            uuid="u1",
                            origin_uuid="u1",
                            uuid_history=["u1"],
                            preferred_label=label,
                            origin_uri="http://example.com",
                            alt_labels=[],
                            description="",
                        ),
                    )
                ],
            )
        ],
        metadata=NELMetadata(nel_model_id="nel-1", taxonomy_model_id="tax-1", processing_time_ms=10.0),
    )


class TestLinkEntities:
    async def test_happy_path_returns_linked_entities(self):
        # GIVEN the service returns a successful response
        svc = FakeNELService(response=_make_response("Head Chef"))

        # WHEN POST /v2/nel is called with valid entities and headers
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.post(
                "/v2/nel",
                json={"entities": [{"text": "Head Chef", "entity_type": "occupation"}]},
                headers={"x-taxonomy-model-id": "tax-1", "x-nel-model-id": "nel-1"},
            )

        # THEN 200 is returned with the linked entity
        assert resp.status_code == 200
        data = resp.json()
        assert data["linked_entities"][0]["matches"][0]["entity"]["preferred_label"] == "Head Chef"

    async def test_forwards_correct_args_to_service(self):
        # GIVEN a fake service that records its call
        svc = FakeNELService(response=_make_response())

        # WHEN POST /v2/nel is called with specific headers and options
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            await client.post(
                "/v2/nel",
                json={"entities": [{"text": "Head Chef", "entity_type": "occupation"}], "top_k": 3, "min_similarity": 0.5},
                headers={"x-taxonomy-model-id": "tax-1", "x-nel-model-id": "nel-1"},
            )

        # THEN the service receives the correct arguments
        assert svc.last_call["taxonomy_model_id"] == "tax-1"
        assert svc.last_call["nel_model_id"] == "nel-1"
        assert svc.last_call["options"] == NELOptions(top_k=3, min_similarity=0.5)
        assert svc.last_call["entities"] == [("Head Chef", EntityType.occupation)]

    async def test_503_when_cache_not_ready(self):
        # GIVEN the service raises EmbeddingsCacheNotReadyError
        svc = FakeNELService(raises=EmbeddingsCacheNotReadyError("tax-1", "nel-1", "generating"))

        # WHEN POST /v2/nel is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.post(
                "/v2/nel",
                json={"entities": [{"text": "Head Chef", "entity_type": "occupation"}]},
                headers={"x-taxonomy-model-id": "tax-1"},
            )

        # THEN 503 is returned
        assert resp.status_code == 503

    async def test_400_when_no_taxonomy_model_id(self):
        # GIVEN no taxonomy model id in headers or env
        svc = FakeNELService(response=_make_response())

        # WHEN POST /v2/nel is called without the taxonomy model id header
        with patch("nel.config.DEFAULT_TAXONOMY_MODEL_ID", ""):
            async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
                resp = await client.post(
                    "/v2/nel",
                    json={"entities": [{"text": "Head Chef", "entity_type": "occupation"}]},
                )

        # THEN 400 is returned
        assert resp.status_code == 400

    async def test_422_when_empty_entities(self):
        # GIVEN a request with an empty entities list
        svc = FakeNELService(response=_make_response())

        # WHEN POST /v2/nel is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.post(
                "/v2/nel",
                json={"entities": []},
                headers={"x-taxonomy-model-id": "tax-1"},
            )

        # THEN Pydantic rejects it with 422
        assert resp.status_code == 422

    async def test_422_when_too_many_entities(self):
        # GIVEN a request with 201 entities (exceeds max_length=200)
        svc = FakeNELService(response=_make_response())
        entities = [{"text": f"entity {i}", "entity_type": "occupation"} for i in range(201)]

        # WHEN POST /v2/nel is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.post(
                "/v2/nel",
                json={"entities": entities},
                headers={"x-taxonomy-model-id": "tax-1"},
            )

        # THEN Pydantic rejects it with 422 before the handler runs
        assert resp.status_code == 422

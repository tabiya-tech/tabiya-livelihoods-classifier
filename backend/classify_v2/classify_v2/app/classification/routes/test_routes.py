"""Tests for classify v2 routes."""

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from classify_v2.app.classification.routes.routes import router, _get_service
from classify_v2.app.classification.service.errors import EmbeddingsCacheNotReadyError, NERServiceError
from classify_v2.app.classification.service.service import IClassifyService
from classify_v2.app.classification.service.types import (
    ClassifiedEntity,
    ClassifyMetadata,
    ClassifyOptions,
    ClassifyResponse,
    EntitySpan,
    OccupationEntity,
    OccupationMatch,
)


class FakeClassifyService(IClassifyService):
    def __init__(self, response: ClassifyResponse | None = None, raises: Exception | None = None):
        self._response = response
        self._raises = raises
        self.last_call: dict | None = None

    async def classify(self, input_text, options=None):
        self.last_call = dict(input_text=input_text, options=options)
        if self._raises:
            raise self._raises
        return self._response


def _make_app(svc: IClassifyService) -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[_get_service] = lambda: svc
    return test_app


def _make_response() -> ClassifyResponse:
    return ClassifyResponse(
        entities=[
            ClassifiedEntity(
                entity_type="occupation",
                surface_form="Head Chef",
                span=EntitySpan(start=0, end=9),
                matches=[
                    OccupationMatch(
                        similarity_score=0.92,
                        entity=OccupationEntity(
                            uuid="u1",
                            origin_uuid="u1",
                            uuid_history=["u1"],
                            preferred_label="Head Chef",
                            origin_uri="http://example.com",
                            alt_labels=[],
                            description="",
                        ),
                    )
                ],
            )
        ],
        metadata=ClassifyMetadata(
            classifier_version="2.0.0",
            ner_model="ner-model",
            nel_model_id="nel-1",
            taxonomy_model_id="tax-1",
            processing_time_ms=100.0,
        ),
    )


class TestClassifyV2:
    async def test_happy_path_returns_200_with_entities(self):
        # GIVEN the service returns a successful response
        svc = FakeClassifyService(response=_make_response())

        # WHEN POST /v2/classify is called with valid text
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.post("/v2/classify", json={"text": "Head Chef needed"})

        # THEN 200 is returned with classified entities
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["entities"]) == 1
        assert data["entities"][0]["matches"][0]["entity"]["preferred_label"] == "Head Chef"

    async def test_input_text_forwarded_to_service(self):
        # GIVEN a service that records its call
        svc = FakeClassifyService(response=_make_response())

        # WHEN POST /v2/classify is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            await client.post("/v2/classify", json={"text": "Head Chef needed"})

        # THEN the input text is passed to the service
        assert svc.last_call["input_text"] == "Head Chef needed"

    async def test_400_when_no_text_provided(self):
        # GIVEN a request with no text, title, or description
        svc = FakeClassifyService(response=_make_response())

        # WHEN POST /v2/classify is called with empty body
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.post("/v2/classify", json={})

        # THEN 400 is returned
        assert resp.status_code == 400

    async def test_title_and_description_accepted(self):
        # GIVEN a request using title + description instead of text
        svc = FakeClassifyService(response=_make_response())

        # WHEN POST /v2/classify is called with title and description
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.post("/v2/classify", json={"title": "Head Chef", "description": "Manage kitchen"})

        # THEN 200 is returned
        assert resp.status_code == 200

    async def test_503_when_embeddings_not_ready(self):
        # GIVEN the service raises EmbeddingsCacheNotReadyError
        svc = FakeClassifyService(raises=EmbeddingsCacheNotReadyError("Embeddings not ready"))

        # WHEN POST /v2/classify is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.post("/v2/classify", json={"text": "Head Chef needed"})

        # THEN 503 is returned
        assert resp.status_code == 503

    async def test_502_when_ner_service_fails(self):
        # GIVEN the NER service is unavailable
        svc = FakeClassifyService(raises=NERServiceError("NER unreachable"))

        # WHEN POST /v2/classify is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.post("/v2/classify", json={"text": "Head Chef needed"})

        # THEN 502 is returned
        assert resp.status_code == 502


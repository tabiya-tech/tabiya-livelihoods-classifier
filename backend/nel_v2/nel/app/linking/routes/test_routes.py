"""Tests for entity linking routes."""

from unittest.mock import patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from nel.app.embedding.service.service import EmbeddingBackendUnavailableError
from nel.app.linking.routes.routes import router, _get_service, _get_user_config
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
from nel.app.user_config.service.types import UserConfig


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


def _make_app(
    svc: INELService,
    user_config: UserConfig | None = None,
) -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router)
    resolved_config = user_config or UserConfig(user_id="user-1", taxonomy_model_id="tax-1", nel_model_id="nel-1")
    test_app.dependency_overrides[_get_service] = lambda: svc
    test_app.dependency_overrides[_get_user_config] = lambda: resolved_config
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
        # GIVEN the service returns a successful response and user has config set
        svc = FakeNELService(response=_make_response("Head Chef"))

        # WHEN POST /v2/nel is called with valid entities
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.post(
                "/v2/nel",
                json={"entities": [{"text": "Head Chef", "entity_type": "occupation"}]},
            )

        # THEN 200 is returned with the linked entity
        assert resp.status_code == 200
        data = resp.json()
        assert data["linked_entities"][0]["matches"][0]["entity"]["preferred_label"] == "Head Chef"

    async def test_forwards_model_ids_from_user_config_to_service(self):
        # GIVEN a user with specific model ids in their config
        svc = FakeNELService(response=_make_response())
        user_config = UserConfig(user_id="user-1", taxonomy_model_id="tax-custom", nel_model_id="nel-custom")

        # WHEN POST /v2/nel is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc, user_config)), base_url="http://test") as client:
            await client.post(
                "/v2/nel",
                json={"entities": [{"text": "Head Chef", "entity_type": "occupation"}], "top_k": 3, "min_similarity": 0.5},
            )

        # THEN the service receives the model ids from the user config
        assert svc.last_call["taxonomy_model_id"] == "tax-custom"
        assert svc.last_call["nel_model_id"] == "nel-custom"
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
            )

        # THEN 503 is returned
        assert resp.status_code == 503

    async def test_503_when_embedding_backend_unavailable(self):
        # GIVEN the embedding backend is unavailable (e.g. Vertex AI disabled)
        svc = FakeNELService(
            raises=EmbeddingBackendUnavailableError(
                "Vertex AI embedding backend unavailable for model 'text-embedding-005'"
            )
        )

        # WHEN POST /v2/nel is called
        async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
            resp = await client.post(
                "/v2/nel",
                json={"entities": [{"text": "Head Chef", "entity_type": "occupation"}]},
            )

        # THEN 503 is returned (not an opaque 500), so the classify pipeline
        # maps it to a retryable upstream-unavailable failure
        assert resp.status_code == 503

    async def test_400_when_user_config_has_no_taxonomy_model_and_no_default(self):
        # GIVEN user has no taxonomy model set and DEFAULT_TAXONOMY_MODEL_ID is empty
        svc = FakeNELService(response=_make_response())
        user_config = UserConfig(user_id="user-1", taxonomy_model_id="", nel_model_id="nel-1")

        # WHEN POST /v2/nel is called
        with patch("nel.config.DEFAULT_TAXONOMY_MODEL_ID", ""):
            async with AsyncClient(transport=ASGITransport(app=_make_app(svc, user_config)), base_url="http://test") as client:
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
            )

        # THEN Pydantic rejects it with 422 before the handler runs
        assert resp.status_code == 422


class TestLanguage:
    async def test_language_uses_its_taxonomy_and_embedding_model(self):
        # GIVEN Spanish is configured with its own taxonomy and embedding model
        given_language = "es"
        given_language_config = {"taxonomy_model_id": "tax-es", "nel_v2_model_id": "nel-es"}
        svc = FakeNELService(response=_make_response())

        # WHEN a Spanish request is linked
        with patch("nel.app.linking.routes.routes.get_language_config", return_value=given_language_config):
            async with AsyncClient(transport=ASGITransport(app=_make_app(svc)), base_url="http://test") as client:
                actual_response = await client.post(
                    "/v2/nel",
                    json={"entities": [{"text": "cocinero", "entity_type": "occupation"}], "language": given_language},
                )

        # THEN the request succeeds
        assert actual_response.status_code == 200
        # AND both models come from the language, so queries match the stored embeddings
        assert svc.last_call["taxonomy_model_id"] == given_language_config["taxonomy_model_id"]
        assert svc.last_call["nel_model_id"] == given_language_config["nel_v2_model_id"]

    async def test_language_without_a_taxonomy_keeps_the_user_pair(self):
        # GIVEN Spanish has an embedding model but no taxonomy model configured
        given_language_config = {"taxonomy_model_id": "", "nel_v2_model_id": "nel-es"}
        given_user_config = UserConfig(user_id="user-1", taxonomy_model_id="tax-user", nel_model_id="nel-user")
        svc = FakeNELService(response=_make_response())

        # WHEN a Spanish request is linked
        with patch("nel.app.linking.routes.routes.get_language_config", return_value=given_language_config):
            async with AsyncClient(
                transport=ASGITransport(app=_make_app(svc, given_user_config)), base_url="http://test"
            ) as client:
                await client.post(
                    "/v2/nel",
                    json={"entities": [{"text": "cocinero", "entity_type": "occupation"}], "language": "es"},
                )

        # THEN the user's taxonomy and embedding model are used together, never mixed with the language's model
        assert svc.last_call["taxonomy_model_id"] == given_user_config.taxonomy_model_id
        assert svc.last_call["nel_model_id"] == given_user_config.nel_model_id

    async def test_no_language_uses_the_user_pair(self):
        # GIVEN a user with their own taxonomy and embedding model
        given_user_config = UserConfig(user_id="user-1", taxonomy_model_id="tax-user", nel_model_id="nel-user")
        svc = FakeNELService(response=_make_response())

        # WHEN a request without a language is linked
        async with AsyncClient(
            transport=ASGITransport(app=_make_app(svc, given_user_config)), base_url="http://test"
        ) as client:
            await client.post("/v2/nel", json={"entities": [{"text": "Head Chef", "entity_type": "occupation"}]})

        # THEN the user's pair is used, as before
        assert svc.last_call["taxonomy_model_id"] == given_user_config.taxonomy_model_id
        assert svc.last_call["nel_model_id"] == given_user_config.nel_model_id

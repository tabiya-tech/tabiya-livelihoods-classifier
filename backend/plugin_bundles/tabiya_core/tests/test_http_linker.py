"""Tests for HttpEntityLinker — the HTTP-backed IEntityLinker.

Uses `httpx.MockTransport` so no real network I/O happens. Every test
uses GIVEN/WHEN/THEN inline comments and named `given*` / `expected*`
variables.
"""

from __future__ import annotations

import httpx
import pytest

from tabiya_core.plugins.nel.core import EmbeddingsCacheNotReady
from tabiya_core.plugins.nel.http_linker import HttpEntityLinker


def _make_async_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_link_posts_to_v2_nel_and_maps_matches():
    # GIVEN a fake NEL v2 service returning one linked entity with two matches
    # in the v2 shape (nested `entity`, `similarity_score`).
    givenBaseUrl = "http://nel-v2-service:5003"
    givenEntities = [("data scientist", "occupation")]
    expectedRequestPath = "/v2/nel"
    expectedV2Response = {
        "linked_entities": [
            {
                "input_text": "data scientist",
                "entity_type": "occupation",
                "matches": [
                    {
                        "entity_type": "occupation",
                        "similarity_score": 0.92,
                        "entity": {
                            "uuid": "u-1",
                            "preferred_label": "data scientist",
                            "origin_uri": "http://data.europa.eu/esco/occupation/xyz",
                            "esco_code": "2529.4",
                        },
                    },
                    {
                        "entity_type": "occupation",
                        "similarity_score": 0.81,
                        "entity": {
                            "uuid": "u-2",
                            "preferred_label": "data analyst",
                            "origin_uri": "http://data.europa.eu/esco/occupation/abc",
                            "esco_code": "2529.5",
                        },
                    },
                ],
            }
        ],
        "metadata": {"nel_model_id": "all-MiniLM-L6-v2", "taxonomy_model_id": "tax-1", "processing_time_ms": 12.3},
    }
    receivedRequests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        receivedRequests.append(request)
        return httpx.Response(200, json=expectedV2Response)

    givenLinker = HttpEntityLinker(
        base_url=givenBaseUrl, http_client=_make_async_client(handler)
    )

    # WHEN the plugin core calls link
    matchesPerEntity = await givenLinker.link(
        entities=givenEntities,
        nel_model_id="all-MiniLM-L6-v2",
        taxonomy_model_id="tax-1",
        top_k=5,
        min_similarity=0.0,
    )

    # THEN one HTTP POST hit /v2/nel
    assert len(receivedRequests) == 1
    assert receivedRequests[0].url.path == expectedRequestPath
    # AND the nested v2 match shape was reshaped into contract Matches
    assert len(matchesPerEntity) == 1
    assert len(matchesPerEntity[0]) == 2
    firstMatch = matchesPerEntity[0][0]
    assert firstMatch.preferred_label == "data scientist"
    assert firstMatch.score == pytest.approx(0.92)
    assert firstMatch.uri == "http://data.europa.eu/esco/occupation/xyz"
    assert firstMatch.id == "2529.4"  # esco_code preferred as the id


@pytest.mark.asyncio
async def test_link_forwards_user_identity_for_model_resolution():
    # GIVEN a user id — nel_v2 resolves models from the authenticated user, so
    # the linker must forward it as the x-apigateway-api-userinfo header.
    import base64
    import json

    givenBaseUrl = "http://nel-v2-service:5003"
    givenUserId = "firebase-uid-123"
    receivedHeaders: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        receivedHeaders.append(request.headers)
        return httpx.Response(200, json={"linked_entities": [], "metadata": {}})

    givenLinker = HttpEntityLinker(
        base_url=givenBaseUrl, http_client=_make_async_client(handler)
    )

    # WHEN link runs with a user_id
    await givenLinker.link(
        entities=[("nurse", "occupation")],
        nel_model_id="any",
        taxonomy_model_id="any",
        top_k=5,
        min_similarity=0.0,
        user_id=givenUserId,
    )

    # THEN the outbound request carries the user identity nel_v2 reads
    header = receivedHeaders[0].get("x-apigateway-api-userinfo")
    assert header is not None
    decoded = json.loads(base64.b64decode(header).decode())
    assert decoded["user_id"] == givenUserId


@pytest.mark.asyncio
async def test_link_returns_empty_list_when_no_entities():
    # GIVEN no entities to link
    givenBaseUrl = "http://nel-service:5003"
    givenNoEntities: list[tuple[str, str]] = []
    receivedRequests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        receivedRequests.append(request)
        return httpx.Response(200, json={"linked_entities": [], "metadata": {}})

    givenLinker = HttpEntityLinker(
        base_url=givenBaseUrl, http_client=_make_async_client(handler)
    )

    # WHEN link runs with an empty batch
    result = await givenLinker.link(
        entities=givenNoEntities,
        nel_model_id="any",
        taxonomy_model_id="any",
        top_k=5,
        min_similarity=0.0,
    )

    # THEN no HTTP request was made and the result is an empty list
    assert result == []
    assert receivedRequests == []


@pytest.mark.asyncio
async def test_link_translates_503_to_embeddings_cache_not_ready():
    # GIVEN a NEL service that returns 503 (embeddings cache warming up)
    givenBaseUrl = "http://nel-service:5003"
    givenEntities = [("data scientist", "occupation")]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="linker not loaded")

    givenLinker = HttpEntityLinker(
        base_url=givenBaseUrl, http_client=_make_async_client(handler)
    )

    # WHEN link runs
    # THEN the 503 is translated to EmbeddingsCacheNotReady so the plugin
    # can map it to UPSTREAM_UNAVAILABLE
    with pytest.raises(EmbeddingsCacheNotReady):
        await givenLinker.link(
            entities=givenEntities,
            nel_model_id="all-MiniLM-L6-v2",
            taxonomy_model_id="esco-v1.2",
            top_k=5,
            min_similarity=0.0,
        )


@pytest.mark.asyncio
async def test_link_forwards_top_k_and_min_similarity():
    # GIVEN a fake NEL service that records the request body
    givenBaseUrl = "http://nel-service:5003"
    givenEntities = [("nurse", "occupation")]
    expectedTopK = 12
    expectedMinSimilarity = 0.42
    receivedBodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        receivedBodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"linked_entities": [{"input_text": "nurse", "entity_type": "occupation", "matches": []}], "metadata": {}})

    givenLinker = HttpEntityLinker(
        base_url=givenBaseUrl, http_client=_make_async_client(handler)
    )

    # WHEN link runs with a specific top_k and min_similarity
    await givenLinker.link(
        entities=givenEntities,
        nel_model_id="any",
        taxonomy_model_id="any",
        top_k=expectedTopK,
        min_similarity=expectedMinSimilarity,
    )

    # THEN the outbound request body carries them at the top level
    # (nel_v2's NELRequest, not the legacy v1 `options` nesting)
    assert receivedBodies[0]["top_k"] == expectedTopK
    assert receivedBodies[0]["min_similarity"] == pytest.approx(expectedMinSimilarity)


def test_constructor_rejects_empty_base_url():
    # GIVEN no base URL configured
    givenEmptyBaseUrl = ""

    # WHEN the linker is constructed
    # THEN it refuses to boot
    with pytest.raises(ValueError):
        HttpEntityLinker(base_url=givenEmptyBaseUrl)

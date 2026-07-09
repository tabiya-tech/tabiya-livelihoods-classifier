"""Tests for HttpEntityExtractor — the HTTP-backed IEntityExtractor.

Uses `httpx.MockTransport` so no real network I/O happens. Every test
uses GIVEN/WHEN/THEN inline comments and named `given*` / `expected*`
variables.
"""

from __future__ import annotations

import httpx
import pytest

from tabiya_core.plugins.ner.http_extractor import HttpEntityExtractor


def _make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_extract_posts_to_v1_ner_and_maps_entities():
    # GIVEN a fake NER service that returns two occupation entities
    givenBaseUrl = "http://ner-service:5002"
    givenText = "Data scientist wanted"
    expectedRequestPath = "/v1/ner"
    expectedEntities = [
        {
            "surface_form": "Data scientist",
            "entity_type": "occupation",
            "span": {"start": 0, "end": 14},
        },
        {
            "surface_form": "wanted",
            "entity_type": "domain",
            "span": {"start": 15, "end": 21},
        },
    ]
    receivedRequests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        receivedRequests.append(request)
        return httpx.Response(200, json={"entities": expectedEntities, "metadata": {}})

    givenExtractor = HttpEntityExtractor(
        base_url=givenBaseUrl, http_client=_make_client(handler)
    )

    # WHEN the plugin core calls extract
    resultEntities = givenExtractor.extract(givenText, model_id="tabiya/roberta-base-job-ner")

    # THEN one HTTP POST hit /v1/ner with the text in the body
    assert len(receivedRequests) == 1
    assert receivedRequests[0].url.path == expectedRequestPath
    assert receivedRequests[0].method == "POST"
    # AND the response entities were mapped into the plugin contract shape
    assert len(resultEntities) == len(expectedEntities)
    assert resultEntities[0].surface_form == expectedEntities[0]["surface_form"]
    assert resultEntities[0].entity_type == expectedEntities[0]["entity_type"]
    assert resultEntities[0].span.start == expectedEntities[0]["span"]["start"]
    assert resultEntities[0].span.end == expectedEntities[0]["span"]["end"]
    assert resultEntities[1].entity_type == expectedEntities[1]["entity_type"]


def test_extract_does_not_forward_entity_types_filter():
    # GIVEN a fake service that echoes any received body back for inspection
    givenBaseUrl = "http://ner-service:5002"
    receivedRequests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        receivedRequests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"entities": [], "metadata": {}})

    givenExtractor = HttpEntityExtractor(
        base_url=givenBaseUrl, http_client=_make_client(handler)
    )

    # WHEN extract is called with a model_id (which is metadata to this adapter)
    givenExtractor.extract("hello", model_id="tabiya/roberta-large-job-ner")

    # THEN the request body contains only `text`; entity_types filtering is
    # owned by the plugin Core and must not leak into the HTTP call.
    expectedRequestKeys = {"text"}
    assert set(receivedRequests[0].keys()) == expectedRequestKeys


def test_extract_raises_on_500_response():
    # GIVEN a fake NER service that returns 500
    givenBaseUrl = "http://ner-service:5002"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    givenExtractor = HttpEntityExtractor(
        base_url=givenBaseUrl, http_client=_make_client(handler)
    )

    # WHEN extract runs
    # THEN httpx.HTTPStatusError bubbles up — the plugin adapter maps this
    # into a PLUGIN_INTERNAL envelope one layer above.
    with pytest.raises(httpx.HTTPStatusError):
        givenExtractor.extract("hello", model_id="any")


def test_constructor_rejects_empty_base_url():
    # GIVEN no base URL configured
    givenEmptyBaseUrl = ""

    # WHEN the extractor is constructed
    # THEN it refuses to boot rather than silently POSTing to a broken URL
    with pytest.raises(ValueError):
        HttpEntityExtractor(base_url=givenEmptyBaseUrl)

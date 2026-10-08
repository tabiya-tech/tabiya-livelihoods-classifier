"""NEL plugin tests — Core + adapter integration.

Uses a fake linker; the real MongoDB-backed linker is wired in at bundle
startup and covered by the existing nel_v2 integration tests.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tabiya_plugin_contracts import (
    Entities,
    Entity,
    EntitySpan,
    Match,
    PluginCategory,
    SlotType,
)
from tabiya_plugin_contracts.adapters.http import make_http_adapter

from tabiya_core.plugins.nel import MANIFEST as NEL_MANIFEST
from tabiya_core.plugins.nel import core as nel_core


class _FakeLinker:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._raise: Exception | None = None
        self._matches_override: list[list[Match]] | None = None
        self._metadata_override: dict | None = None

    def with_matches(self, matches: list[list[Match]]) -> "_FakeLinker":
        self._matches_override = matches
        return self

    def with_metadata(self, metadata: dict) -> "_FakeLinker":
        self._metadata_override = metadata
        return self

    def with_raise(self, exc: Exception) -> "_FakeLinker":
        self._raise = exc
        return self

    async def link(
        self,
        entities: list[tuple[str, str]],
        nel_model_id: str,
        taxonomy_model_id: str,
        top_k: int,
        min_similarity: float,
        user_id: str | None = None,
    ) -> tuple[list[list[Match]], dict]:
        self.calls.append(
            {
                "entities": entities,
                "nel_model_id": nel_model_id,
                "taxonomy_model_id": taxonomy_model_id,
                "top_k": top_k,
                "min_similarity": min_similarity,
                "user_id": user_id,
            }
        )
        if self._raise is not None:
            raise self._raise
        metadata = self._metadata_override or {}
        if self._matches_override is not None:
            return self._matches_override, metadata
        matches = [
            [
                Match(
                    id=f"esco/{entity_type}/{surface}",
                    preferred_label=surface,
                    score=0.9,
                    uri=f"http://taxonomy.tabiya.tech/{entity_type}/{surface}",
                )
            ]
            for surface, entity_type in entities
        ]
        return matches, metadata


@pytest.fixture
def fake_linker():
    givenLinker = _FakeLinker()
    nel_core.set_linker(givenLinker)
    yield givenLinker
    nel_core._linker = None


def _client() -> TestClient:
    app = FastAPI()
    router = make_http_adapter(NEL_MANIFEST, nel_core.invoke)
    app.include_router(router, prefix=f"/plugin/{NEL_MANIFEST.plugin_id}")
    return TestClient(app)


def _entities_payload(items: list[tuple[str, str]], source_text: str = "src") -> dict:
    return {
        "entities": [
            {
                "surface_form": surface,
                "entity_type": entity_type,
                "span": {"start": index, "end": index + len(surface)},
            }
            for index, (surface, entity_type) in enumerate(items)
        ],
        "source_text": source_text,
    }


def _invoke_body(
    entities: list[tuple[str, str]],
    config: dict | None = None,
    source_text: str = "sample source",
) -> dict:
    return {
        "context": {"request_id": "req-1", "stage_index": 1, "deadline_ms": 45_000},
        "config": config
        or {
            "nel_model_id": "all-MiniLM-L6-v2",
            "taxonomy_model_id": "model-abc",
        },
        "input": _entities_payload(entities, source_text=source_text),
    }


def test_manifest_declares_entities_input_and_linked_entities_output() -> None:
    # GIVEN the NEL manifest
    givenManifest = NEL_MANIFEST

    # THEN it links Entities -> LinkedEntities
    assert givenManifest.input_slot.type == SlotType.ENTITIES
    assert givenManifest.output_slot.type == SlotType.LINKED_ENTITIES
    assert givenManifest.category == PluginCategory.CORE



def test_invoke_links_each_linkable_entity_via_the_linker(fake_linker) -> None:
    # GIVEN two linkable entities
    givenEntities = [("Statistician", "occupation"), ("Python", "skill")]
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{NEL_MANIFEST.plugin_id}/invoke", json=_invoke_body(givenEntities)
    )

    # THEN both entities come back linked with the fake's default match
    expectedStatus = 200
    expectedEntityCount = 2
    assert response.status_code == expectedStatus
    body = response.json()
    linked = body["output"]["entities"]
    assert len(linked) == expectedEntityCount
    assert linked[0]["matches"][0]["preferred_label"] == "Statistician"
    assert linked[1]["matches"][0]["preferred_label"] == "Python"


def test_invoke_surfaces_backend_resolved_model_ids_in_metadata(fake_linker) -> None:
    # GIVEN the linking backend resolves models from the user's config and
    # reports them back (the stage config here carries different ids)
    fake_linker.with_metadata(
        {"nel_model_id": "resolved-nel", "taxonomy_model_id": "resolved-tax"}
    )
    givenEntities = [("Statistician", "occupation")]
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{NEL_MANIFEST.plugin_id}/invoke", json=_invoke_body(givenEntities)
    )

    # THEN the invoke metadata reports the ids the backend actually resolved,
    # not the stage config's — so classify shows real ids, never "unknown"
    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert metadata["nel_model_id"] == "resolved-nel"
    assert metadata["taxonomy_model_id"] == "resolved-tax"



def test_invoke_passes_non_linkable_entities_through_with_empty_matches(fake_linker) -> None:
    # GIVEN one linkable (skill) and one non-linkable (experience) entity
    givenEntities = [
        ("Python", "skill"),
        ("5 years", "experience"),
    ]
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{NEL_MANIFEST.plugin_id}/invoke", json=_invoke_body(givenEntities)
    )

    # THEN the linker only saw the skill, and experience passes through with []
    expectedStatus = 200
    assert response.status_code == expectedStatus
    linked = response.json()["output"]["entities"]
    assert len(fake_linker.calls) == 1
    assert fake_linker.calls[0]["entities"] == [("Python", "skill")]
    assert linked[0]["matches"] != []
    assert linked[1]["matches"] == []


def test_invoke_with_no_linkable_entities_skips_linker_call(fake_linker) -> None:
    # GIVEN only non-linkable entity types
    givenEntities = [("some domain", "domain"), ("5 yrs", "experience")]
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{NEL_MANIFEST.plugin_id}/invoke", json=_invoke_body(givenEntities)
    )

    # THEN the linker was never invoked and every entity carries []
    expectedStatus = 200
    expectedCallCount = 0
    assert response.status_code == expectedStatus
    assert len(fake_linker.calls) == expectedCallCount
    assert all(entity["matches"] == [] for entity in response.json()["output"]["entities"])


def test_invoke_returns_empty_linked_when_input_has_no_entities(fake_linker) -> None:
    # GIVEN an empty entities payload
    givenEntities: list[tuple[str, str]] = []
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{NEL_MANIFEST.plugin_id}/invoke", json=_invoke_body(givenEntities)
    )

    # THEN we get an empty entities list without calling the linker
    expectedStatus = 200
    assert response.status_code == expectedStatus
    assert response.json()["output"]["entities"] == []
    assert len(fake_linker.calls) == 0


def test_invoke_maps_embeddings_cache_not_ready_to_upstream_unavailable_503(
    fake_linker,
) -> None:
    # GIVEN a linker that raises EmbeddingsCacheNotReady
    fake_linker.with_raise(
        nel_core.EmbeddingsCacheNotReady(
            taxonomy_model_id="model-abc",
            nel_model_id="all-MiniLM-L6-v2",
            current_status="processing",
        )
    )
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{NEL_MANIFEST.plugin_id}/invoke",
        json=_invoke_body([("Statistician", "occupation")]),
    )

    # THEN the adapter surfaces UPSTREAM_UNAVAILABLE 503
    expectedStatus = 503
    assert response.status_code == expectedStatus
    assert response.json()["code"] == "UPSTREAM_UNAVAILABLE"



def test_invoke_with_linker_returning_wrong_length_maps_to_bad_input(fake_linker) -> None:
    # GIVEN a linker that returns the wrong number of match-lists
    fake_linker.with_matches([])
    client = _client()
    givenEntities = [("Python", "skill")]

    # WHEN we invoke
    response = client.post(
        f"/plugin/{NEL_MANIFEST.plugin_id}/invoke", json=_invoke_body(givenEntities)
    )

    # THEN the plugin refuses with BAD_INPUT (linker contract violation)
    expectedStatus = 400
    assert response.status_code == expectedStatus
    assert response.json()["code"] == "BAD_INPUT"

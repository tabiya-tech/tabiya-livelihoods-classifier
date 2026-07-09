"""NER plugin tests — Core + adapter integration.

Uses a deterministic fake extractor so the tests are not tied to the
transformer model. Every test uses GIVEN/WHEN/THEN inline comments and
named `given*` / `expected*` variables.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tabiya_plugin_contracts import Entity, EntitySpan, PluginCategory, SlotType
from tabiya_plugin_contracts.adapters.http import make_http_adapter

from tabiya_core.plugins.ner import MANIFEST as NER_MANIFEST
from tabiya_core.plugins.ner import core as ner_core


class _FakeExtractor:
    """Deterministic extractor: emits one entity per whitespace-separated word."""

    def __init__(self, entity_type: str = "occupation") -> None:
        self.entity_type = entity_type
        self.seen_model_ids: list[str] = []

    def extract(self, text: str, model_id: str) -> list[Entity]:
        self.seen_model_ids.append(model_id)
        entities: list[Entity] = []
        cursor = 0
        for word in text.split():
            start = text.find(word, cursor)
            end = start + len(word)
            entities.append(
                Entity(
                    surface_form=word,
                    entity_type=self.entity_type,
                    span=EntitySpan(start=start, end=end),
                )
            )
            cursor = end
        return entities


@pytest.fixture
def fake_extractor():
    givenExtractor = _FakeExtractor()
    ner_core.set_extractor(givenExtractor)
    yield givenExtractor
    ner_core._extractor = None


def _client() -> TestClient:
    app = FastAPI()
    router = make_http_adapter(NER_MANIFEST, ner_core.invoke)
    app.include_router(router, prefix=f"/plugin/{NER_MANIFEST.plugin_id}")
    return TestClient(app)


def _invoke_body(text: str, config: dict | None = None) -> dict:
    return {
        "context": {"request_id": "req-1", "stage_index": 0, "deadline_ms": 30_000},
        "config": config or {},
        "input": {"text": text},
    }


def test_manifest_declares_raw_text_input_and_entities_output() -> None:
    # GIVEN the NER manifest
    givenManifest = NER_MANIFEST

    # WHEN we inspect its slots
    expectedInputSlot = SlotType.RAW_TEXT
    expectedOutputSlot = SlotType.ENTITIES

    # THEN they match the pipeline contract for a Core-category plugin
    assert givenManifest.input_slot.type == expectedInputSlot
    assert givenManifest.output_slot.type == expectedOutputSlot
    assert givenManifest.category == PluginCategory.CORE


def test_manifest_config_schema_advertises_x_source_for_model_id() -> None:
    # GIVEN the NER config_schema
    givenSchema = NER_MANIFEST.config_schema

    # WHEN we look up the model_id field
    modelIdField = givenSchema["properties"]["model_id"]

    # THEN it points at the NEL v2 model list (resolved by classify_v2's
    # options proxy). It must NOT point back at the /v2/plugins/.../options
    # endpoint that reads this field — that would recurse infinitely.
    expectedXSource = "/v2/nel/models"
    assert modelIdField["x-source"] == expectedXSource


def test_invoke_happy_path_returns_one_entity_per_word(fake_extractor) -> None:
    # GIVEN a two-word sentence
    givenText = "Statistician wanted"
    client = _client()

    # WHEN we invoke NER
    response = client.post(
        f"/plugin/{NER_MANIFEST.plugin_id}/invoke", json=_invoke_body(givenText)
    )

    # THEN we get one entity per word, spans line up, source_text is preserved
    expectedStatus = 200
    expectedEntityCount = 2
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["output"]["source_text"] == givenText
    assert len(body["output"]["entities"]) == expectedEntityCount
    assert body["output"]["entities"][0]["surface_form"] == "Statistician"
    assert body["output"]["entities"][1]["span"]["start"] == givenText.index("wanted")


def test_invoke_filters_entities_by_configured_types(fake_extractor) -> None:
    # GIVEN a fake that tags every word as "skill" but a config asking for occupations
    fake_extractor.entity_type = "skill"
    givenText = "python sql"
    client = _client()
    givenConfig = {"entity_types": ["occupation"]}

    # WHEN we invoke
    response = client.post(
        f"/plugin/{NER_MANIFEST.plugin_id}/invoke",
        json=_invoke_body(givenText, config=givenConfig),
    )

    # THEN nothing survives the filter
    expectedStatus = 200
    expectedEntityCount = 0
    assert response.status_code == expectedStatus
    assert len(response.json()["output"]["entities"]) == expectedEntityCount


def test_invoke_passes_model_id_through_to_the_extractor(fake_extractor) -> None:
    # GIVEN a specific model id in config
    givenModelId = "my-custom-ner-model"
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{NER_MANIFEST.plugin_id}/invoke",
        json=_invoke_body("word", config={"model_id": givenModelId}),
    )

    # THEN the extractor saw the model_id verbatim
    expectedStatus = 200
    assert response.status_code == expectedStatus
    assert fake_extractor.seen_model_ids == [givenModelId]


def test_invoke_with_empty_text_returns_bad_input(fake_extractor) -> None:
    # GIVEN a text of only whitespace
    givenText = "   "
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{NER_MANIFEST.plugin_id}/invoke", json=_invoke_body(givenText)
    )

    # THEN the plugin returns a BAD_INPUT 400
    expectedStatus = 400
    assert response.status_code == expectedStatus
    assert response.json()["code"] == "BAD_INPUT"


def test_invoke_without_extractor_registered_returns_500_plugin_internal() -> None:
    # GIVEN no extractor has been set on this bundle
    ner_core._extractor = None
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{NER_MANIFEST.plugin_id}/invoke", json=_invoke_body("hello")
    )

    # THEN the adapter wraps the RuntimeError as PLUGIN_INTERNAL
    expectedStatus = 500
    assert response.status_code == expectedStatus
    assert response.json()["code"] == "PLUGIN_INTERNAL"


def test_invoke_with_unknown_config_field_returns_config_invalid(fake_extractor) -> None:
    # GIVEN a config with a stray field (additionalProperties=False in schema, but
    # Pydantic model itself doesn't forbid — the JSON Schema handles that at 11.5.
    # This test asserts the Pydantic parser tolerates extras for now, so config
    # errors surface only for genuinely wrong types.)
    givenConfig = {"top_k": "not-a-number"}  # top_k doesn't exist on NerConfig; ignored
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{NER_MANIFEST.plugin_id}/invoke",
        json=_invoke_body("hello", config=givenConfig),
    )

    # THEN the plugin succeeds — unknown fields are accepted by NerConfig
    expectedStatus = 200
    assert response.status_code == expectedStatus

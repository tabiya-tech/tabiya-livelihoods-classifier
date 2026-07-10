"""JSON-entities source plugin tests."""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tabiya_plugin_contracts import PluginCategory, SlotType
from tabiya_plugin_contracts.adapters.http import make_http_adapter

from tabiya_io.plugins.json_entities import MANIFEST as JSON_ENTITIES_MANIFEST
from tabiya_io.plugins.json_entities import invoke as json_entities_invoke


def _client() -> TestClient:
    app = FastAPI()
    router = make_http_adapter(JSON_ENTITIES_MANIFEST, json_entities_invoke)
    app.include_router(router, prefix=f"/plugin/{JSON_ENTITIES_MANIFEST.plugin_id}")
    return TestClient(app)


def _invoke_body(config: dict) -> dict:
    return {
        "context": {"request_id": "req-1", "stage_index": 0, "deadline_ms": 5_000},
        "config": config,
        "input": {"kind": "None"},
    }


def test_manifest_declares_none_input_and_entities_output() -> None:
    # GIVEN the json_entities manifest
    givenManifest = JSON_ENTITIES_MANIFEST

    # THEN it's a Source that emits Entities (so it can feed NEL directly)
    assert givenManifest.category == PluginCategory.SOURCE
    assert givenManifest.input_slot.type == SlotType.NONE
    assert givenManifest.output_slot.type == SlotType.ENTITIES


def test_invoke_builds_entities_from_json_array() -> None:
    # GIVEN a JSON array of pre-extracted occupations
    givenConfig = {
        "json": json.dumps(
            [
                {"surface_form": "head chef", "entity_type": "occupation"},
                {"surface_form": "welding", "entity_type": "skill"},
            ]
        ),
        "source_text": "from the ops database",
    }
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{JSON_ENTITIES_MANIFEST.plugin_id}/invoke",
        json=_invoke_body(givenConfig),
    )

    # THEN the output is an Entities payload with both entities + source text
    expectedStatus = 200
    assert response.status_code == expectedStatus
    output = response.json()["output"]
    assert output["source_text"] == "from the ops database"
    assert [entity["surface_form"] for entity in output["entities"]] == [
        "head chef",
        "welding",
    ]
    assert output["entities"][0]["entity_type"] == "occupation"


def test_invoke_defaults_span_when_absent() -> None:
    # GIVEN entities without spans (pre-extracted, no offsets)
    givenConfig = {"json": json.dumps([{"surface_form": "nurse", "entity_type": "occupation"}])}
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{JSON_ENTITIES_MANIFEST.plugin_id}/invoke",
        json=_invoke_body(givenConfig),
    )

    # THEN a zero span is filled in
    assert response.status_code == 200
    span = response.json()["output"]["entities"][0]["span"]
    assert span == {"start": 0, "end": 0}


def test_invoke_with_item_missing_fields_returns_config_invalid() -> None:
    # GIVEN an entity missing entity_type
    givenConfig = {"json": json.dumps([{"surface_form": "x"}])}
    expectedStatus = 400
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{JSON_ENTITIES_MANIFEST.plugin_id}/invoke",
        json=_invoke_body(givenConfig),
    )

    # THEN it's a clean config error
    assert response.status_code == expectedStatus


def test_invoke_with_non_array_returns_config_invalid() -> None:
    # GIVEN a JSON object instead of an array
    givenConfig = {"json": json.dumps({"surface_form": "x", "entity_type": "y"})}
    expectedStatus = 400
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{JSON_ENTITIES_MANIFEST.plugin_id}/invoke",
        json=_invoke_body(givenConfig),
    )

    # THEN the non-array case is rejected
    assert response.status_code == expectedStatus

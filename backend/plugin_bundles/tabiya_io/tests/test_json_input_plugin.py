"""JSON-input source plugin tests."""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tabiya_plugin_contracts import PluginCategory, SlotType
from tabiya_plugin_contracts.adapters.http import make_http_adapter

from tabiya_io.plugins.json_input import MANIFEST as JSON_INPUT_MANIFEST
from tabiya_io.plugins.json_input import invoke as json_input_invoke


def _client() -> TestClient:
    app = FastAPI()
    router = make_http_adapter(JSON_INPUT_MANIFEST, json_input_invoke)
    app.include_router(router, prefix=f"/plugin/{JSON_INPUT_MANIFEST.plugin_id}")
    return TestClient(app)


def _invoke_body(config: dict) -> dict:
    return {
        "context": {"request_id": "req-1", "stage_index": 0, "deadline_ms": 5_000},
        "config": config,
        "input": {"kind": "None"},
    }


def test_manifest_declares_none_input_and_raw_text_output() -> None:
    # GIVEN the json_input manifest
    givenManifest = JSON_INPUT_MANIFEST

    # THEN it has the Source shape: None → RawText
    assert givenManifest.category == PluginCategory.SOURCE
    assert givenManifest.input_slot.type == SlotType.NONE
    assert givenManifest.output_slot.type == SlotType.RAW_TEXT


def test_invoke_reads_default_text_field() -> None:
    # GIVEN a JSON payload with the default 'text' field
    givenText = "Head chef wanted for busy kitchen."
    givenConfig = {"json": json.dumps({"text": givenText})}
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{JSON_INPUT_MANIFEST.plugin_id}/invoke",
        json=_invoke_body(givenConfig),
    )

    # THEN the default 'text' field becomes the output
    expectedStatus = 200
    assert response.status_code == expectedStatus
    assert response.json()["output"] == {"text": givenText}


def test_invoke_reads_custom_text_field() -> None:
    # GIVEN a payload whose body lives in a 'description' field
    givenDescription = "Analyse survey data for the research team."
    givenConfig = {
        "json": json.dumps({"description": givenDescription, "title": "Data Scientist"}),
        "text_field": "description",
    }
    client = _client()

    # WHEN we invoke pointing text_field at 'description'
    response = client.post(
        f"/plugin/{JSON_INPUT_MANIFEST.plugin_id}/invoke",
        json=_invoke_body(givenConfig),
    )

    # THEN the chosen field's value is the output text
    expectedStatus = 200
    assert response.status_code == expectedStatus
    assert response.json()["output"]["text"] == givenDescription


def test_invoke_with_invalid_json_returns_config_invalid() -> None:
    # GIVEN a payload that isn't valid JSON
    givenConfig = {"json": "{not json"}
    expectedStatus = 400
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{JSON_INPUT_MANIFEST.plugin_id}/invoke",
        json=_invoke_body(givenConfig),
    )

    # THEN it's a config error (400), not a 500
    assert response.status_code == expectedStatus


def test_invoke_with_missing_field_returns_config_invalid() -> None:
    # GIVEN valid JSON but the target field is absent
    givenConfig = {"json": json.dumps({"headline": "x"}), "text_field": "text"}
    expectedStatus = 400
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{JSON_INPUT_MANIFEST.plugin_id}/invoke",
        json=_invoke_body(givenConfig),
    )

    # THEN the missing-field case is a clean config error
    assert response.status_code == expectedStatus

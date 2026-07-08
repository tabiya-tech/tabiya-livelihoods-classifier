"""Text-input source plugin tests."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tabiya_plugin_contracts import PluginCategory, SlotType
from tabiya_plugin_contracts.adapters.http import make_http_adapter

from tabiya_io.plugins.text_input import MANIFEST as TEXT_INPUT_MANIFEST
from tabiya_io.plugins.text_input import invoke as text_input_invoke


def _client() -> TestClient:
    app = FastAPI()
    router = make_http_adapter(TEXT_INPUT_MANIFEST, text_input_invoke)
    app.include_router(router, prefix=f"/plugin/{TEXT_INPUT_MANIFEST.plugin_id}")
    return TestClient(app)


def _invoke_body(config: dict) -> dict:
    return {
        "context": {"request_id": "req-1", "stage_index": 0, "deadline_ms": 5_000},
        "config": config,
        "input": {"kind": "None"},
    }


def test_manifest_declares_none_input_and_raw_text_output() -> None:
    # GIVEN the text_input manifest
    givenManifest = TEXT_INPUT_MANIFEST

    # THEN slots match the Source shape
    assert givenManifest.category == PluginCategory.SOURCE
    assert givenManifest.input_slot.type == SlotType.NONE
    assert givenManifest.output_slot.type == SlotType.RAW_TEXT


def test_invoke_with_text_returns_text_verbatim() -> None:
    # GIVEN a text config
    givenText = "Statistician needed for research team."
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{TEXT_INPUT_MANIFEST.plugin_id}/invoke",
        json=_invoke_body({"text": givenText}),
    )

    # THEN the output carries the exact text
    expectedStatus = 200
    assert response.status_code == expectedStatus
    assert response.json()["output"] == {"text": givenText}


def test_invoke_with_title_and_description_joins_with_blank_line() -> None:
    # GIVEN a job ad shape
    givenTitle = "Data Scientist"
    givenDescription = "Analyse survey data."
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{TEXT_INPUT_MANIFEST.plugin_id}/invoke",
        json=_invoke_body({"title": givenTitle, "description": givenDescription}),
    )

    # THEN they're joined by a blank line so NER sees two paragraphs
    expectedStatus = 200
    expectedText = f"{givenTitle}\n\n{givenDescription}"
    assert response.status_code == expectedStatus
    assert response.json()["output"]["text"] == expectedText


def test_invoke_with_only_description_still_returns_text() -> None:
    # GIVEN just a description
    givenDescription = "Some job description."
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{TEXT_INPUT_MANIFEST.plugin_id}/invoke",
        json=_invoke_body({"description": givenDescription}),
    )

    # THEN the description alone becomes the output text
    expectedStatus = 200
    assert response.status_code == expectedStatus
    assert response.json()["output"]["text"] == givenDescription


def test_invoke_with_empty_config_returns_config_invalid() -> None:
    # GIVEN a config with nothing populated
    givenConfig: dict = {}
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{TEXT_INPUT_MANIFEST.plugin_id}/invoke",
        json=_invoke_body(givenConfig),
    )

    # THEN we get CONFIG_INVALID
    expectedStatus = 400
    assert response.status_code == expectedStatus
    assert response.json()["code"] == "CONFIG_INVALID"


def test_invoke_with_text_and_title_together_returns_config_invalid() -> None:
    # GIVEN both shapes provided at once — ambiguous, should be rejected
    givenConfig = {"text": "one", "title": "two"}
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{TEXT_INPUT_MANIFEST.plugin_id}/invoke",
        json=_invoke_body(givenConfig),
    )

    # THEN CONFIG_INVALID
    expectedStatus = 400
    assert response.status_code == expectedStatus
    assert response.json()["code"] == "CONFIG_INVALID"

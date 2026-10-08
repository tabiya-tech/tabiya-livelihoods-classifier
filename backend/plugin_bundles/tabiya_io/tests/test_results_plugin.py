"""Results sink plugin tests."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tabiya_plugin_contracts import PluginCategory, SlotType
from tabiya_plugin_contracts.adapters.http import make_http_adapter

from tabiya_io.plugins.results import MANIFEST as RESULTS_MANIFEST
from tabiya_io.plugins.results import invoke as results_invoke


def _client() -> TestClient:
    app = FastAPI()
    router = make_http_adapter(RESULTS_MANIFEST, results_invoke)
    app.include_router(router, prefix=f"/plugin/{RESULTS_MANIFEST.plugin_id}")
    return TestClient(app)


def _linked_payload() -> dict:
    return {
        "entities": [
            {
                "surface_form": "Statistician",
                "entity_type": "occupation",
                "span": {"start": 0, "end": 12},
                "matches": [
                    {
                        "id": "esco/occupation/statistician",
                        "preferred_label": "statistician",
                        "score": 0.91,
                        "uri": "http://taxonomy.tabiya.tech/occupation/statistician",
                    }
                ],
            }
        ],
        "source_text": "Statistician wanted.",
    }


def _invoke_body(config: dict | None = None) -> dict:
    return {
        "context": {"request_id": "req-1", "stage_index": 3, "deadline_ms": 5_000},
        "config": config or {},
        "input": _linked_payload(),
    }


def test_manifest_declares_linked_entities_input_and_none_output() -> None:
    # GIVEN the results sink manifest
    givenManifest = RESULTS_MANIFEST

    # THEN slots match the Sink shape
    assert givenManifest.category == PluginCategory.SINK
    assert givenManifest.input_slot.type == SlotType.LINKED_ENTITIES
    assert givenManifest.output_slot.type == SlotType.NONE


def test_invoke_happy_path_returns_none_sentinel() -> None:
    # GIVEN a valid linked entities payload
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{RESULTS_MANIFEST.plugin_id}/invoke", json=_invoke_body()
    )

    # THEN we get a NoneSlot back
    expectedStatus = 200
    expectedOutput = {"kind": "None"}
    assert response.status_code == expectedStatus
    assert response.json()["output"] == expectedOutput


def test_invoke_with_extra_config_field_returns_config_invalid() -> None:
    # GIVEN an unknown config field
    givenConfig = {"format": "csv"}
    client = _client()

    # WHEN we invoke
    response = client.post(
        f"/plugin/{RESULTS_MANIFEST.plugin_id}/invoke", json=_invoke_body(givenConfig)
    )

    # THEN CONFIG_INVALID surfaces
    expectedStatus = 400
    assert response.status_code == expectedStatus
    assert response.json()["code"] == "CONFIG_INVALID"

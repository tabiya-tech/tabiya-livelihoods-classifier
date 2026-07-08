"""Bundle-level smoke tests for tabiya_io.

Mirrors the tabiya_core suite; kept as its own test file so each bundle's
pytest run stays self-contained (bundles ship as independent artifacts).
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from tabiya_plugin_contracts import (
    Context,
    Manifest,
    NoneSlot,
    PluginCategory,
    RawText,
    Slot,
    SlotType,
)
from tabiya_plugin_contracts.adapters.auth import require_identity_token
from tabiya_plugin_contracts.adapters.http import make_http_adapter


@pytest.fixture(autouse=True)
def _force_local_mode(monkeypatch):
    monkeypatch.setenv("TARGET_ENVIRONMENT_TYPE", "local")
    yield


def _source_manifest(plugin_id: str = "test.source.text.v1") -> Manifest:
    # A minimal Source-shaped manifest: input_slot = None, output_slot = RawText.
    return Manifest(
        plugin_id=plugin_id,
        name="Text Input",
        version="0.1.0",
        category=PluginCategory.SOURCE,
        summary="Emits static text.",
        icon="text",
        input_slot=Slot(type=SlotType.NONE, cardinality="none"),
        output_slot=Slot(type=SlotType.RAW_TEXT),
        config_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        timeout_ms=5_000,
    )


async def _static_source(input: NoneSlot, config: dict, context: Context) -> RawText:
    return RawText(text=config.get("text", "hello"))


def _build_app(installed: list[tuple[Manifest, object, object | None]]) -> FastAPI:
    app = FastAPI()
    for manifest, invoke_fn, health_fn in installed:
        router = make_http_adapter(manifest, invoke_fn, health_fn)
        app.include_router(
            router,
            prefix=f"/plugin/{manifest.plugin_id}",
            dependencies=[Depends(require_identity_token)],
        )

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "bundle": "tabiya_io",
            "plugin_count": len(installed),
            "installed": [manifest.plugin_id for manifest, _, _ in installed],
        }

    return app


def test_io_bundle_health_reflects_installed_source_plugin() -> None:
    # GIVEN a bundle with a single source-shaped plugin installed
    givenManifest = _source_manifest()
    client = TestClient(_build_app([(givenManifest, _static_source, None)]))

    # WHEN we hit /health
    response = client.get("/health")

    # THEN it lists the plugin id
    expectedStatus = 200
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["installed"] == [givenManifest.plugin_id]


def test_io_bundle_invoke_of_source_returns_raw_text() -> None:
    # GIVEN a source plugin configured with static text
    givenManifest = _source_manifest()
    client = TestClient(_build_app([(givenManifest, _static_source, None)]))
    givenText = "job description here"

    # WHEN we invoke with an empty input (None slot) and config carrying the text
    response = client.post(
        f"/plugin/{givenManifest.plugin_id}/invoke",
        json={
            "context": {"request_id": "req-1", "stage_index": 0, "deadline_ms": 5_000},
            "config": {"text": givenText},
            "input": {"kind": "None"},
        },
    )

    # THEN the output payload matches the RawText slot with the expected text
    expectedStatus = 200
    assert response.status_code == expectedStatus
    assert response.json()["output"] == {"text": givenText}


def test_io_bundle_returns_404_for_unknown_plugin() -> None:
    # GIVEN an empty bundle
    client = TestClient(_build_app([]))

    # WHEN we ask for a plugin
    response = client.get("/plugin/not.installed.v1/manifest")

    # THEN 404
    expectedStatus = 404
    assert response.status_code == expectedStatus

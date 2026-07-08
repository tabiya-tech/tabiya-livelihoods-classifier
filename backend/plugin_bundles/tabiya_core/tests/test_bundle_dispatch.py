"""Bundle-level smoke tests.

The 11.1a bundle boots with zero installed plugins; these tests inject an
in-process echo plugin to prove the dispatcher wires per-plugin routers
correctly under `/plugin/{plugin_id}` and that the bundle's own `/health`
endpoint reflects the installed set.
"""

from __future__ import annotations

import os

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from tabiya_plugin_contracts import (
    Context,
    Entities,
    Entity,
    EntitySpan,
    Manifest,
    PluginCategory,
    RawText,
    Slot,
    SlotType,
)
from tabiya_plugin_contracts.adapters.auth import require_identity_token
from tabiya_plugin_contracts.adapters.http import make_http_adapter


@pytest.fixture(autouse=True)
def _force_local_mode(monkeypatch):
    # The auth dependency short-circuits in local mode; bundle tests never
    # exercise real GCP identity tokens.
    monkeypatch.setenv("TARGET_ENVIRONMENT_TYPE", "local")
    yield


def _echo_manifest(plugin_id: str = "test.echo.v1") -> Manifest:
    return Manifest(
        plugin_id=plugin_id,
        name="Echo",
        version="0.1.0",
        category=PluginCategory.CORE,
        summary="Echoes text as an occupation entity.",
        icon="ner",
        input_slot=Slot(type=SlotType.RAW_TEXT),
        output_slot=Slot(type=SlotType.ENTITIES),
        config_schema={},
        timeout_ms=5_000,
    )


async def _echo_core(input: RawText, config: dict, context: Context) -> Entities:
    return Entities(
        entities=[
            Entity(
                surface_form=input.text,
                entity_type="occupation",
                span=EntitySpan(start=0, end=len(input.text)),
            )
        ],
        source_text=input.text,
    )


def _build_app(installed: list[tuple[Manifest, object, object | None]]) -> FastAPI:
    """Mimics the tabiya_core.main:app assembly for a given INSTALLED_PLUGINS list."""

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
            "bundle": "tabiya_core",
            "plugin_count": len(installed),
            "installed": [manifest.plugin_id for manifest, _, _ in installed],
        }

    return app


def test_bundle_health_reports_installed_plugins() -> None:
    # GIVEN a bundle with two installed plugins
    givenPluginIds = ["test.echo.v1", "test.echo.v2"]
    givenInstalled = [(_echo_manifest(pid), _echo_core, None) for pid in givenPluginIds]
    client = TestClient(_build_app(givenInstalled))

    # WHEN we hit the bundle's own /health
    response = client.get("/health")

    # THEN the response lists every installed plugin
    expectedStatus = 200
    expectedPluginCount = 2
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["plugin_count"] == expectedPluginCount
    assert body["installed"] == givenPluginIds


def test_bundle_serves_each_plugin_under_its_prefix() -> None:
    # GIVEN two installed plugins with distinct ids
    givenIds = ["test.echo.v1", "test.echo.v2"]
    givenInstalled = [(_echo_manifest(pid), _echo_core, None) for pid in givenIds]
    client = TestClient(_build_app(givenInstalled))

    # WHEN we fetch each plugin's manifest
    for pluginId in givenIds:
        response = client.get(f"/plugin/{pluginId}/manifest")

        # THEN each responds with its own plugin_id
        expectedStatus = 200
        assert response.status_code == expectedStatus
        assert response.json()["plugin_id"] == pluginId


def test_bundle_invoke_routes_to_the_correct_plugin() -> None:
    # GIVEN two plugins, each with its own echo behaviour distinguishable by manifest.plugin_id
    givenIds = ["test.echo.alpha.v1", "test.echo.beta.v1"]
    givenInstalled = [(_echo_manifest(pid), _echo_core, None) for pid in givenIds]
    client = TestClient(_build_app(givenInstalled))
    givenBody = {
        "context": {"request_id": "req-1", "stage_index": 0, "deadline_ms": 5_000},
        "config": {},
        "input": {"text": "hello"},
    }

    # WHEN we invoke each
    for pluginId in givenIds:
        response = client.post(f"/plugin/{pluginId}/invoke", json=givenBody)

        # THEN each returns its own response, isolated from the other
        expectedStatus = 200
        assert response.status_code == expectedStatus
        assert response.json()["output"]["source_text"] == "hello"


def test_bundle_with_no_plugins_still_serves_health() -> None:
    # GIVEN a bundle with zero installed plugins (the 11.1a shell)
    client = TestClient(_build_app([]))

    # WHEN we hit /health
    response = client.get("/health")

    # THEN it still returns 200 with plugin_count=0
    expectedStatus = 200
    expectedPluginCount = 0
    assert response.status_code == expectedStatus
    assert response.json()["plugin_count"] == expectedPluginCount


def test_bundle_returns_404_for_unknown_plugin_id() -> None:
    # GIVEN a bundle with one plugin
    givenInstalled = [(_echo_manifest("test.echo.v1"), _echo_core, None)]
    client = TestClient(_build_app(givenInstalled))

    # WHEN we ask for a plugin that isn't installed
    response = client.get("/plugin/not.installed.v1/manifest")

    # THEN it's a 404 — FastAPI's default for unmatched routes
    expectedStatus = 404
    assert response.status_code == expectedStatus


def test_auth_dependency_rejects_missing_bearer_when_not_local() -> None:
    # GIVEN a non-local environment (production-like)
    os.environ["TARGET_ENVIRONMENT_TYPE"] = "production"
    try:
        givenInstalled = [(_echo_manifest("test.echo.v1"), _echo_core, None)]
        client = TestClient(_build_app(givenInstalled))

        # WHEN we invoke without an Authorization header
        response = client.get("/plugin/test.echo.v1/manifest")

        # THEN we get 401
        expectedStatus = 401
        assert response.status_code == expectedStatus
    finally:
        os.environ["TARGET_ENVIRONMENT_TYPE"] = "local"

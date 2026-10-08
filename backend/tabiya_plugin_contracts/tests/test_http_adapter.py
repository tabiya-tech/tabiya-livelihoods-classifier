"""Tests for the shared HTTP adapter helper.

Exercises the contract surface without any real plugin: a tiny synchronous
Core function that echoes RawText → Entities is enough to prove the wire
behaviour (parsing, validation, error mapping, timeout, contract-version
emission).
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tabiya_plugin_contracts import (
    CONTRACT_VERSION,
    Context,
    Entities,
    Entity,
    EntitySpan,
    ErrorCode,
    Health,
    HealthStatus,
    Manifest,
    PluginCategory,
    RawText,
    Slot,
    SlotType,
)
from tabiya_plugin_contracts.adapters.http import (
    BadInputError,
    ConfigInvalidError,
    UpstreamUnavailableError,
    make_http_adapter,
)


def _echo_manifest(timeout_ms: int = 5_000) -> Manifest:
    return Manifest(
        plugin_id="test.echo.v1",
        name="Echo",
        version="0.1.0",
        category=PluginCategory.CORE,
        summary="Echoes input text back as a single entity.",
        icon="ner",
        input_slot=Slot(type=SlotType.RAW_TEXT),
        output_slot=Slot(type=SlotType.ENTITIES),
        config_schema={},
        timeout_ms=timeout_ms,
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


def _client_for(
    manifest: Manifest,
    invoke_fn=_echo_core,
    health_fn=None,
) -> TestClient:
    app = FastAPI()
    router = make_http_adapter(manifest, invoke_fn, health_fn)
    app.include_router(router, prefix=f"/plugin/{manifest.plugin_id}")
    return TestClient(app)


def _invoke_body(text: str = "Statistician") -> dict:
    return {
        "context": {
            "request_id": "req-1",
            "user_id": "user-a",
            "pipeline_id": "pipe-1",
            "stage_index": 0,
            "deadline_ms": 5_000,
        },
        "config": {},
        "input": {"text": text},
    }


def test_manifest_endpoint_emits_contract_version() -> None:
    # GIVEN a manifest without an explicit contract-version field
    givenManifest = _echo_manifest()
    client = _client_for(givenManifest)

    # WHEN a client fetches the manifest
    response = client.get(f"/plugin/{givenManifest.plugin_id}/manifest")

    # THEN the adapter has injected the contract version under its aliased key
    expectedStatus = 200
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["x-tabiya-contract-version"] == CONTRACT_VERSION
    assert body["plugin_id"] == givenManifest.plugin_id


def test_invoke_happy_path_returns_typed_output() -> None:
    # GIVEN an echo plugin
    givenManifest = _echo_manifest()
    client = _client_for(givenManifest)
    givenText = "Statistician wanted"

    # WHEN we invoke it
    response = client.post(
        f"/plugin/{givenManifest.plugin_id}/invoke",
        json=_invoke_body(givenText),
    )

    # THEN the output slot matches the declared Entities type
    expectedStatus = 200
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["output"]["source_text"] == givenText
    expectedEntityCount = 1
    assert len(body["output"]["entities"]) == expectedEntityCount
    assert body["output"]["entities"][0]["surface_form"] == givenText


def test_invoke_with_wrong_input_shape_returns_bad_input_envelope() -> None:
    # GIVEN a request whose input does not match the RawText slot
    givenManifest = _echo_manifest()
    client = _client_for(givenManifest)
    givenBody = _invoke_body()
    givenBody["input"] = {"not_text": 42}

    # WHEN we invoke
    response = client.post(
        f"/plugin/{givenManifest.plugin_id}/invoke", json=givenBody
    )

    # THEN we get a 400 with a BAD_INPUT envelope
    expectedStatus = 400
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["code"] == ErrorCode.BAD_INPUT.value


def test_invoke_with_malformed_json_returns_bad_input_envelope() -> None:
    # GIVEN a request body that isn't JSON
    givenManifest = _echo_manifest()
    client = _client_for(givenManifest)

    # WHEN we send raw bytes
    response = client.post(
        f"/plugin/{givenManifest.plugin_id}/invoke",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )

    # THEN the adapter returns 400 BAD_INPUT rather than crashing
    expectedStatus = 400
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["code"] == ErrorCode.BAD_INPUT.value


def test_invoke_that_exceeds_timeout_returns_504() -> None:
    # GIVEN a manifest with a 100ms timeout and a Core that sleeps longer
    givenManifest = _echo_manifest(timeout_ms=100)

    async def slowCore(input: RawText, config: dict, context: Context) -> Entities:
        await asyncio.sleep(0.5)
        return await _echo_core(input, config, context)

    client = _client_for(givenManifest, invoke_fn=slowCore)

    # WHEN we invoke
    response = client.post(
        f"/plugin/{givenManifest.plugin_id}/invoke", json=_invoke_body()
    )

    # THEN the adapter returns 504 TIMEOUT
    expectedStatus = 504
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["code"] == ErrorCode.TIMEOUT.value


def test_invoke_that_raises_upstream_error_maps_to_503() -> None:
    # GIVEN a Core that raises UpstreamUnavailableError
    givenManifest = _echo_manifest()

    async def upstreamDown(input: RawText, config: dict, context: Context) -> Entities:
        raise UpstreamUnavailableError("embeddings cache not ready")

    client = _client_for(givenManifest, invoke_fn=upstreamDown)

    # WHEN we invoke
    response = client.post(
        f"/plugin/{givenManifest.plugin_id}/invoke", json=_invoke_body()
    )

    # THEN the adapter surfaces the mapped code + status
    expectedStatus = 503
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["code"] == ErrorCode.UPSTREAM_UNAVAILABLE.value


def test_invoke_that_raises_config_invalid_maps_to_400() -> None:
    # GIVEN a Core that rejects its config
    givenManifest = _echo_manifest()

    async def rejectsConfig(input: RawText, config: dict, context: Context) -> Entities:
        raise ConfigInvalidError("model_id missing")

    client = _client_for(givenManifest, invoke_fn=rejectsConfig)

    # WHEN we invoke
    response = client.post(
        f"/plugin/{givenManifest.plugin_id}/invoke", json=_invoke_body()
    )

    # THEN we get CONFIG_INVALID with 400
    expectedStatus = 400
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["code"] == ErrorCode.CONFIG_INVALID.value


def test_invoke_unhandled_exception_maps_to_plugin_internal_500() -> None:
    # GIVEN a Core that raises a plain exception
    givenManifest = _echo_manifest()

    async def blowsUp(input: RawText, config: dict, context: Context) -> Entities:
        raise RuntimeError("kaboom")

    client = _client_for(givenManifest, invoke_fn=blowsUp)

    # WHEN we invoke
    response = client.post(
        f"/plugin/{givenManifest.plugin_id}/invoke", json=_invoke_body()
    )

    # THEN it becomes a PLUGIN_INTERNAL 500 envelope, not a stack trace
    expectedStatus = 500
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["code"] == ErrorCode.PLUGIN_INTERNAL.value


def test_invoke_output_mismatch_maps_to_plugin_internal_500() -> None:
    # GIVEN a Core that returns something that isn't valid for the output slot
    givenManifest = _echo_manifest()

    async def wrongShape(input: RawText, config: dict, context: Context):
        return {"totally": "wrong"}

    client = _client_for(givenManifest, invoke_fn=wrongShape)

    # WHEN we invoke
    response = client.post(
        f"/plugin/{givenManifest.plugin_id}/invoke", json=_invoke_body()
    )

    # THEN it's PLUGIN_INTERNAL 500 — the plugin, not the caller, is at fault
    expectedStatus = 500
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["code"] == ErrorCode.PLUGIN_INTERNAL.value


def test_health_defaults_to_ok_when_no_health_fn_provided() -> None:
    # GIVEN an adapter without an explicit health function
    givenManifest = _echo_manifest()
    client = _client_for(givenManifest)

    # WHEN we hit /health
    response = client.get(f"/plugin/{givenManifest.plugin_id}/health")

    # THEN default is "ok"
    expectedStatus = 200
    assert response.status_code == expectedStatus
    assert response.json()["status"] == HealthStatus.OK.value


def test_health_reports_degraded_when_provided_fn_returns_degraded() -> None:
    # GIVEN a plugin whose health function reports degraded
    givenManifest = _echo_manifest()

    async def degradedHealth() -> Health:
        return Health(status=HealthStatus.DEGRADED, detail="model warming up")

    client = _client_for(givenManifest, health_fn=degradedHealth)

    # WHEN we hit /health
    response = client.get(f"/plugin/{givenManifest.plugin_id}/health")

    # THEN it surfaces the degraded status verbatim
    expectedStatus = 200
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["status"] == HealthStatus.DEGRADED.value
    assert body["detail"] == "model warming up"


def test_health_survives_health_fn_raising() -> None:
    # GIVEN a health function that raises
    givenManifest = _echo_manifest()

    async def brokenHealth() -> Health:
        raise RuntimeError("health check itself crashed")

    client = _client_for(givenManifest, health_fn=brokenHealth)

    # WHEN we hit /health
    response = client.get(f"/plugin/{givenManifest.plugin_id}/health")

    # THEN /health still returns 200 with status=down, so orchestrator can route
    expectedStatus = 200
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["status"] == HealthStatus.DOWN.value


def test_bad_input_envelope_carries_validation_error_detail() -> None:
    # GIVEN a request whose envelope itself is malformed (missing context)
    givenManifest = _echo_manifest()
    client = _client_for(givenManifest)

    # WHEN we invoke without a context field
    response = client.post(
        f"/plugin/{givenManifest.plugin_id}/invoke",
        json={"config": {}, "input": {"text": "foo"}},
    )

    # THEN we get BAD_INPUT with the pydantic errors in detail
    expectedStatus = 400
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["code"] == ErrorCode.BAD_INPUT.value
    assert "errors" in body["detail"]

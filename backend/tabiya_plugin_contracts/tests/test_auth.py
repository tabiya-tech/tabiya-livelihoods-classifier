"""Identity-token dependency tests.

Focus on the two non-GCP-touching paths: the local-mode bypass and the
request-audience derivation that avoids a Pulumi self-reference cycle. The
actual token verification delegates to google-auth and is not exercised
here (it needs a real signed token + network).
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException, Request

from tabiya_plugin_contracts.adapters.auth import (
    _audience_from_request,
    require_identity_token,
)


def _make_request(headers: dict[str, str]) -> Request:
    """Build a minimal ASGI Request carrying the given headers."""

    rawHeaders = [
        (name.lower().encode(), value.encode()) for name, value in headers.items()
    ]
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/plugin/tabiya.ner.v1/invoke",
        "headers": rawHeaders,
        "scheme": "http",
        "server": ("tabiya-core", 5010),
    }
    return Request(scope)


def test_audience_from_request_uses_forwarded_proto_and_host():
    # GIVEN a request forwarded by Cloud Run with proto + host headers
    givenHeaders = {"host": "tabiya-core-abc.run.app", "x-forwarded-proto": "https"}
    expectedAudience = "https://tabiya-core-abc.run.app"

    # WHEN the audience is derived from the request
    actualAudience = _audience_from_request(_make_request(givenHeaders))

    # THEN it reconstructs the bundle's own base URL
    assert actualAudience == expectedAudience


def test_audience_from_request_none_when_host_absent():
    # GIVEN a request with no Host header
    givenHeaders: dict[str, str] = {}
    expectedAudience = None

    # WHEN the audience is derived
    actualAudience = _audience_from_request(_make_request(givenHeaders))

    # THEN no audience can be reconstructed
    assert actualAudience is expectedAudience


async def test_require_identity_token_bypasses_in_local_mode(monkeypatch):
    # GIVEN local mode is active
    monkeypatch.setenv("TARGET_ENVIRONMENT_TYPE", "local")
    givenRequest = _make_request({"host": "tabiya-core:5010"})

    # WHEN the dependency runs with no Authorization header
    # THEN it returns without raising (auth is bypassed locally)
    await require_identity_token(givenRequest, authorization=None)


async def test_require_identity_token_rejects_missing_bearer(monkeypatch):
    # GIVEN production mode (not local)
    monkeypatch.setenv("TARGET_ENVIRONMENT_TYPE", "dev")
    givenRequest = _make_request({"host": "tabiya-core-abc.run.app"})
    expectedStatus = 401

    # WHEN the dependency runs with no bearer token
    with pytest.raises(HTTPException) as raised:
        await require_identity_token(givenRequest, authorization=None)

    # THEN it rejects with 401
    assert raised.value.status_code == expectedStatus

"""Identity-token provider tests.

The audience derivation is pure, so it's tested directly. The token fetch
delegates to google-auth via a worker thread; we substitute the static
fetch helper so no GCP metadata server is touched.
"""

from __future__ import annotations

import pytest

from classify_v2.app.pipelines.executor.identity import (
    GcpIdentityTokenProvider,
    _audience_for,
)


def test_audience_strips_path_and_query():
    # GIVEN a full plugin invoke URL with a path and query string
    givenInvokeUrl = "https://tabiya-core-abc123.run.app/plugin/tabiya.ner.v1/invoke?x=1"
    # AND the audience Cloud Run expects is only scheme + host
    expectedAudience = "https://tabiya-core-abc123.run.app"

    # WHEN the audience is derived
    actualAudience = _audience_for(givenInvokeUrl)

    # THEN the path and query are dropped
    assert actualAudience == expectedAudience


def test_audience_preserves_local_host_and_port():
    # GIVEN a local bundle URL with an explicit port
    givenInvokeUrl = "http://tabiya-core:5010/plugin/tabiya.ner.v1/invoke"
    # AND the audience keeps scheme, host, and port
    expectedAudience = "http://tabiya-core:5010"

    # WHEN the audience is derived
    actualAudience = _audience_for(givenInvokeUrl)

    # THEN scheme, host and port survive
    assert actualAudience == expectedAudience


async def test_get_id_token_returns_minted_token(monkeypatch):
    # GIVEN a provider whose underlying google-auth fetch returns a token
    givenToken = "eyJhbGciOi.header.signature"
    givenInvokeUrl = "https://tabiya-io-xyz.run.app/plugin/tabiya.sink.results.v1/invoke"
    expectedAudience = "https://tabiya-io-xyz.run.app"
    seenAudiences: list[str] = []

    def fakeFetch(audience: str) -> str:
        seenAudiences.append(audience)
        return givenToken

    provider = GcpIdentityTokenProvider()
    monkeypatch.setattr(provider, "_fetch_id_token", staticmethod(fakeFetch))

    # WHEN a token is requested for the invoke URL
    actualToken = await provider.get_id_token(givenInvokeUrl)

    # THEN the minted token is returned and the audience was the service URL
    assert actualToken == givenToken
    assert seenAudiences == [expectedAudience]


async def test_get_id_token_swallows_errors(monkeypatch):
    # GIVEN a provider whose fetch raises (e.g. metadata server unreachable)
    givenInvokeUrl = "https://tabiya-core-abc.run.app/plugin/tabiya.ner.v1/invoke"
    expectedToken = None

    def raisingFetch(audience: str) -> str:
        raise RuntimeError("metadata server unreachable")

    provider = GcpIdentityTokenProvider()
    monkeypatch.setattr(provider, "_fetch_id_token", staticmethod(raisingFetch))

    # WHEN a token is requested
    actualToken = await provider.get_id_token(givenInvokeUrl)

    # THEN the failure is swallowed and no token is returned (request proceeds)
    assert actualToken is expectedToken

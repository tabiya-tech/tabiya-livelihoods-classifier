"""Tests for fetching peer OpenAPI schemas into the unified /docs."""

import asyncio
from http import HTTPStatus
from unittest.mock import patch

from fastapi.testclient import TestClient

from classify.main import app
from classify.openapi_merge import PeerSchemaCache, PeerSpec


def _peer(name: str) -> PeerSpec:
    return PeerSpec(name=name, base_url=f"http://{name}.internal", tag=f"{name} tag", prefix=name.upper())


def _peer_schema(path: str) -> dict:
    return {"openapi": "3.1.0", "paths": {path: {"get": {"operationId": path.strip("/")}}}}


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestPeerSchemaCache:
    async def test_missing_peer_is_fetched_once_the_retry_interval_has_passed(self):
        # GIVEN a peer whose first fetch fails and second fetch succeeds
        given_peer = _peer("ner")
        given_schema = _peer_schema("/v1/ner")
        given_clock = FakeClock()
        given_retry_interval_seconds = 30.0
        peer_cache = PeerSchemaCache([given_peer], retry_interval_seconds=given_retry_interval_seconds, clock=given_clock)

        with patch("classify.openapi_merge.fetch_peer_schema", side_effect=[None, given_schema]):
            # AND the startup fetch already failed
            assert await peer_cache.fetch_missing() is False

            # WHEN the retry interval passes and the missing peer is fetched again
            given_clock.now += given_retry_interval_seconds
            got_new_schema = await peer_cache.fetch_missing()

        # THEN a new schema is reported
        assert got_new_schema is True
        # AND the peer's schema is cached
        assert peer_cache.schemas == {given_peer.name: given_schema}
        # AND no peers are missing any more
        assert peer_cache.missing_peers() == []

    async def test_missing_peer_is_not_fetched_again_within_the_retry_interval(self):
        # GIVEN a peer that is unreachable
        given_clock = FakeClock()
        given_retry_interval_seconds = 30.0
        peer_cache = PeerSchemaCache([_peer("ner")], retry_interval_seconds=given_retry_interval_seconds, clock=given_clock)

        with patch("classify.openapi_merge.fetch_peer_schema", return_value=None) as fetch_peer_schema:
            # AND it was just attempted
            await peer_cache.fetch_missing()

            # WHEN another fetch is requested before the retry interval passes
            given_clock.now += given_retry_interval_seconds - 1
            got_new_schema = await peer_cache.fetch_missing()

        # THEN nothing new is reported
        assert got_new_schema is False
        # AND the peer was only fetched by the first attempt
        assert fetch_peer_schema.call_count == 1

    async def test_only_missing_peers_are_fetched_again(self):
        # GIVEN two peers where only the first answers at startup
        given_fetched_peer = _peer("classify_v2")
        given_missing_peer = _peer("nel_v2")
        given_fetched_schema = _peer_schema("/v2/classify")
        given_missing_schema = _peer_schema("/v2/nel")
        given_clock = FakeClock()
        given_retry_interval_seconds = 30.0
        peer_cache = PeerSchemaCache(
            [given_fetched_peer, given_missing_peer],
            retry_interval_seconds=given_retry_interval_seconds,
            clock=given_clock,
        )
        fetched_peer_names: list[str] = []
        startup_results = {given_fetched_peer.name: given_fetched_schema, given_missing_peer.name: None}

        async def fake_fetch_peer_schema(peer: PeerSpec):
            fetched_peer_names.append(peer.name)
            if len(fetched_peer_names) <= len(startup_results):
                return startup_results[peer.name]
            return given_missing_schema

        with patch("classify.openapi_merge.fetch_peer_schema", side_effect=fake_fetch_peer_schema):
            await peer_cache.fetch_missing()

            # WHEN the retry interval passes and missing peers are fetched again
            given_clock.now += given_retry_interval_seconds
            await peer_cache.fetch_missing()

        # THEN the retry only asked the missing peer
        assert fetched_peer_names[len(startup_results):] == [given_missing_peer.name]
        # AND both schemas are cached
        assert peer_cache.schemas == {
            given_fetched_peer.name: given_fetched_schema,
            given_missing_peer.name: given_missing_schema,
        }

    async def test_concurrent_callers_share_one_attempt(self):
        # GIVEN a peer that is slow to answer
        given_schema = _peer_schema("/v1/ner")
        peer_cache = PeerSchemaCache([_peer("ner")], clock=FakeClock())

        async def slow_fetch_peer_schema(_peer_spec: PeerSpec):
            await asyncio.sleep(0.01)
            return given_schema

        with patch("classify.openapi_merge.fetch_peer_schema", side_effect=slow_fetch_peer_schema) as fetch_peer_schema:
            # WHEN two callers fetch missing peers at the same time
            first_result, second_result = await asyncio.gather(peer_cache.fetch_missing(), peer_cache.fetch_missing())

        # THEN the peer was fetched once
        assert fetch_peer_schema.call_count == 1
        # AND exactly one caller reports the new schema
        assert sorted([first_result, second_result]) == [False, True]


class TestUnifiedOpenApiRoute:
    def test_peer_missing_at_startup_appears_once_it_answers(self):
        # GIVEN a peer that was unreachable when the instance started
        given_peer = _peer("nel_v2")
        given_peer_path = "/v2/nel"
        given_clock = FakeClock()
        given_retry_interval_seconds = 30.0
        peer_cache = PeerSchemaCache([given_peer], retry_interval_seconds=given_retry_interval_seconds, clock=given_clock)
        original_peer_cache = getattr(app.state, "openapi_peer_cache", None)
        app.state.openapi_peer_cache = peer_cache
        app.openapi_schema = None
        client = TestClient(app)

        try:
            with patch("classify.openapi_merge.fetch_peer_schema", return_value=None):
                # AND the docs were already served without it
                first_response = client.get("/openapi.json")
            assert given_peer_path not in first_response.json()["paths"]

            # WHEN the peer answers after the retry interval and the docs are requested again
            given_clock.now += given_retry_interval_seconds
            with patch("classify.openapi_merge.fetch_peer_schema", return_value=_peer_schema(given_peer_path)):
                second_response = client.get("/openapi.json")
        finally:
            app.state.openapi_peer_cache = original_peer_cache
            app.openapi_schema = None

        # THEN the docs are served
        assert second_response.status_code == HTTPStatus.OK
        # AND they now include the peer's path under its tag
        assert second_response.json()["paths"][given_peer_path]["get"]["tags"] == [given_peer.tag]

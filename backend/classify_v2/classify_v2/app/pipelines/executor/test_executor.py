"""Pipeline executor tests.

Uses a fake `httpx.AsyncClient` and a stub registry so no network / real
plugin bundles are touched.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx
import pytest
from tabiya_plugin_contracts import (
    CONTRACT_VERSION,
    Manifest,
    PluginCategory,
    Slot,
    SlotType,
)

from classify_v2.app.pipelines.executor import (
    PipelineExecutor,
    PluginInvocationError,
    PluginTimeoutError,
    PluginUpstreamUnavailableError,
)
from classify_v2.app.pipelines.registry import (
    PluginRegistry,
    PluginStatus,
    ResolvedPlugin,
)
from classify_v2.app.pipelines.repository import PipelineDocument, StageDocument


CORE_URL = "http://tabiya-core:5010"
IO_URL = "http://tabiya-io:5011"


class _FakeAsyncClient:
    """Async httpx-compatible test double.

    Configure with `on(url, response)` or `raise_on(url, exc)`. The
    executor only uses `.post(url, json=..., headers=..., timeout=...)`.
    """

    def __init__(self) -> None:
        self._responses: dict[str, Callable[[dict[str, Any]], httpx.Response]] = {}
        self._exceptions: dict[str, Exception] = {}
        self.calls: list[dict[str, Any]] = []

    def on(self, url: str, response: httpx.Response) -> "_FakeAsyncClient":
        self._responses[url] = lambda body: response
        return self

    def on_dynamic(
        self,
        url: str,
        factory: Callable[[dict[str, Any]], httpx.Response],
    ) -> "_FakeAsyncClient":
        self._responses[url] = factory
        return self

    def raise_on(self, url: str, exc: Exception) -> "_FakeAsyncClient":
        self._exceptions[url] = exc
        return self

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        self.calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        if url in self._exceptions:
            raise self._exceptions[url]
        if url in self._responses:
            return self._responses[url](json or {})
        return httpx.Response(404, text="not configured for test")


def _manifest(
    plugin_id: str,
    *,
    category: PluginCategory,
    input_slot: SlotType,
    output_slot: SlotType,
    timeout_ms: int = 5_000,
) -> Manifest:
    return Manifest(
        plugin_id=plugin_id,
        name=plugin_id,
        version="0.1.0",
        category=category,
        summary="test",
        icon="ner",
        input_slot=Slot(type=input_slot, cardinality="none" if input_slot == SlotType.NONE else "single"),
        output_slot=Slot(type=output_slot, cardinality="none" if output_slot == SlotType.NONE else "single"),
        config_schema={},
        timeout_ms=timeout_ms,
        **{"x-tabiya-contract-version": CONTRACT_VERSION},
    )


def _resolved(manifest: Manifest, *, base_url: str) -> ResolvedPlugin:
    return ResolvedPlugin(
        plugin_id=manifest.plugin_id,
        resolved_url=f"{base_url}/plugin/{manifest.plugin_id}",
        manifest=manifest,
        status=PluginStatus.ENABLED,
    )


class _StubRegistry:
    def __init__(self, entries: dict[str, ResolvedPlugin]) -> None:
        self._entries = entries

    def get(self, plugin_id: str) -> Optional[ResolvedPlugin]:
        return self._entries.get(plugin_id)


def _canonical_registry() -> _StubRegistry:
    return _StubRegistry(
        {
            "tabiya.source.text.v1": _resolved(
                _manifest(
                    "tabiya.source.text.v1",
                    category=PluginCategory.SOURCE,
                    input_slot=SlotType.NONE,
                    output_slot=SlotType.RAW_TEXT,
                ),
                base_url=IO_URL,
            ),
            "tabiya.ner.v1": _resolved(
                _manifest(
                    "tabiya.ner.v1",
                    category=PluginCategory.CORE,
                    input_slot=SlotType.RAW_TEXT,
                    output_slot=SlotType.ENTITIES,
                ),
                base_url=CORE_URL,
            ),
            "tabiya.nel.v1": _resolved(
                _manifest(
                    "tabiya.nel.v1",
                    category=PluginCategory.CORE,
                    input_slot=SlotType.ENTITIES,
                    output_slot=SlotType.LINKED_ENTITIES,
                ),
                base_url=CORE_URL,
            ),
            "tabiya.sink.results.v1": _resolved(
                _manifest(
                    "tabiya.sink.results.v1",
                    category=PluginCategory.SINK,
                    input_slot=SlotType.LINKED_ENTITIES,
                    output_slot=SlotType.NONE,
                ),
                base_url=IO_URL,
            ),
        }
    )


def _canonical_pipeline() -> PipelineDocument:
    now = datetime.now(timezone.utc)
    return PipelineDocument(
        pipeline_id="pipe-1",
        user_id="uid-1",
        name="Test Pipeline",
        stages=[
            StageDocument(plugin_id="tabiya.source.text.v1", config={"text": ""}),
            StageDocument(plugin_id="tabiya.ner.v1", config={}),
            StageDocument(
                plugin_id="tabiya.nel.v1",
                config={
                    "nel_model_id": "m",
                    "taxonomy_model_id": "t",
                    "top_k": 5,
                    "min_similarity": 0.0,
                },
            ),
            StageDocument(plugin_id="tabiya.sink.results.v1", config={}),
        ],
        is_active=True,
        is_default=False,
        is_readonly=False,
        created_at=now,
        updated_at=now,
    )


def _plugin_url(plugin_id: str) -> str:
    base = CORE_URL if plugin_id in {"tabiya.ner.v1", "tabiya.nel.v1"} else IO_URL
    return f"{base}/plugin/{plugin_id}/invoke"


def _wire_happy_path(http: _FakeAsyncClient) -> None:
    """Configure the fake HTTP to return the canonical stage outputs."""

    def text_response(body: dict[str, Any]) -> httpx.Response:
        text = body.get("config", {}).get("text", "")
        return httpx.Response(
            200,
            json={
                "output": {"text": text},
                "metadata": {"processing_time_ms": 1.0},
            },
        )

    def ner_response(body: dict[str, Any]) -> httpx.Response:
        text = body["input"].get("text", "")
        return httpx.Response(
            200,
            json={
                "output": {
                    "entities": [
                        {
                            "surface_form": text or "job",
                            "entity_type": "occupation",
                            "span": {"start": 0, "end": len(text) or 3},
                        }
                    ],
                    "source_text": text,
                },
                "metadata": {"model_name": "test-ner", "processing_time_ms": 2.0},
            },
        )

    def nel_response(body: dict[str, Any]) -> httpx.Response:
        entities = body["input"].get("entities", [])
        source_text = body["input"].get("source_text", "")
        return httpx.Response(
            200,
            json={
                "output": {
                    "entities": [
                        {
                            **entity,
                            "matches": [
                                {
                                    "id": f"esco/{entity['entity_type']}/{entity['surface_form']}",
                                    "preferred_label": entity["surface_form"],
                                    "score": 0.9,
                                    "uri": None,
                                }
                            ],
                        }
                        for entity in entities
                    ],
                    "source_text": source_text,
                },
                "metadata": {"nel_model_id": "m", "taxonomy_model_id": "t"},
            },
        )

    def results_response(body: dict[str, Any]) -> httpx.Response:
        return httpx.Response(200, json={"output": {"kind": "None"}, "metadata": None})

    http.on_dynamic(_plugin_url("tabiya.source.text.v1"), text_response)
    http.on_dynamic(_plugin_url("tabiya.ner.v1"), ner_response)
    http.on_dynamic(_plugin_url("tabiya.nel.v1"), nel_response)
    http.on_dynamic(_plugin_url("tabiya.sink.results.v1"), results_response)


def _executor(http: _FakeAsyncClient, registry: _StubRegistry | None = None) -> PipelineExecutor:
    return PipelineExecutor(
        registry=(registry or _canonical_registry()),  # type: ignore[arg-type]
        http_client=http,  # type: ignore[arg-type]
    )


async def test_run_calls_every_stage_in_order() -> None:
    # GIVEN a canonical four-stage pipeline
    givenPipeline = _canonical_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_happy_path(fake_http)
    executor = _executor(fake_http)

    # WHEN we run it
    await executor.run(
        pipeline=givenPipeline,
        source_overrides={"text": "Statistician wanted"},
        request_id="req-1",
        user_id="uid-1",
    )

    # THEN each plugin's /invoke was hit exactly once, in stage order
    expectedUrls = [
        _plugin_url("tabiya.source.text.v1"),
        _plugin_url("tabiya.ner.v1"),
        _plugin_url("tabiya.nel.v1"),
        _plugin_url("tabiya.sink.results.v1"),
    ]
    assert [call["url"] for call in fake_http.calls] == expectedUrls


async def test_run_returns_four_stage_outcomes_with_ok_status() -> None:
    # GIVEN the canonical happy path
    givenPipeline = _canonical_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_happy_path(fake_http)
    executor = _executor(fake_http)

    # WHEN we run it
    result = await executor.run(
        pipeline=givenPipeline,
        source_overrides={"text": "job"},
        request_id="req-1",
        user_id="uid-1",
    )

    # THEN we get one StageOutcome per stage, all "ok", with categories set
    expectedCategories = ["source", "core", "core", "sink"]
    assert [outcome.status for outcome in result.stages] == ["ok"] * 4
    assert [outcome.category for outcome in result.stages] == expectedCategories
    assert [outcome.plugin_id for outcome in result.stages] == [
        "tabiya.source.text.v1",
        "tabiya.ner.v1",
        "tabiya.nel.v1",
        "tabiya.sink.results.v1",
    ]


async def test_source_overrides_merge_into_stage_zero_config() -> None:
    # GIVEN a persisted source stage with empty text
    givenPipeline = _canonical_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_happy_path(fake_http)
    executor = _executor(fake_http)
    givenText = "Statistician wanted for research team."

    # WHEN we run with a source override
    await executor.run(
        pipeline=givenPipeline,
        source_overrides={"text": givenText},
        request_id="req-1",
        user_id="uid-1",
    )

    # THEN the source stage's config carried the override to the plugin
    source_call = fake_http.calls[0]
    assert source_call["json"]["config"]["text"] == givenText


async def test_stage_output_becomes_next_stage_input() -> None:
    # GIVEN the canonical happy path
    givenPipeline = _canonical_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_happy_path(fake_http)
    executor = _executor(fake_http)

    # WHEN we run it
    await executor.run(
        pipeline=givenPipeline,
        source_overrides={"text": "job"},
        request_id="req-1",
        user_id="uid-1",
    )

    # THEN the NER stage saw the source's RawText, and NEL saw NER's Entities
    ner_call = fake_http.calls[1]
    nel_call = fake_http.calls[2]
    assert ner_call["json"]["input"] == {"text": "job"}
    assert "entities" in nel_call["json"]["input"]
    assert nel_call["json"]["input"]["entities"][0]["surface_form"] == "job"


async def test_run_snapshots_last_linked_entities_payload() -> None:
    # GIVEN the canonical happy path
    givenPipeline = _canonical_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_happy_path(fake_http)
    executor = _executor(fake_http)

    # WHEN we run it
    result = await executor.run(
        pipeline=givenPipeline,
        source_overrides={"text": "job"},
        request_id="req-1",
        user_id="uid-1",
    )

    # THEN linked_entities_payload contains the NEL output
    assert result.linked_entities_payload is not None
    assert result.linked_entities_payload["source_text"] == "job"
    assert result.linked_entities_payload["entities"][0]["matches"][0]["preferred_label"] == "job"


async def test_run_forwards_pipeline_and_stage_index_in_context() -> None:
    # GIVEN the canonical happy path
    givenPipeline = _canonical_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_happy_path(fake_http)
    executor = _executor(fake_http)

    # WHEN we run
    await executor.run(
        pipeline=givenPipeline,
        source_overrides={"text": "job"},
        request_id="req-42",
        user_id="uid-7",
    )

    # THEN each invoke body's context matches the stage
    for stage_index, call in enumerate(fake_http.calls):
        context = call["json"]["context"]
        assert context["request_id"] == "req-42"
        assert context["user_id"] == "uid-7"
        assert context["pipeline_id"] == givenPipeline.pipeline_id
        assert context["stage_index"] == stage_index


async def test_run_sends_contract_version_header_on_every_stage() -> None:
    # GIVEN the canonical happy path
    givenPipeline = _canonical_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_happy_path(fake_http)
    executor = _executor(fake_http)

    # WHEN we run
    await executor.run(
        pipeline=givenPipeline,
        source_overrides={"text": "job"},
        request_id="req-1",
        user_id="uid-1",
    )

    # THEN every request carried the contract version header
    for call in fake_http.calls:
        assert call["headers"]["x-tabiya-contract-version"] == CONTRACT_VERSION


async def test_run_uses_manifest_timeout_ms_for_each_stage() -> None:
    # GIVEN a pipeline where NER's manifest has an unusual timeout
    custom_ner = _manifest(
        "tabiya.ner.v1",
        category=PluginCategory.CORE,
        input_slot=SlotType.RAW_TEXT,
        output_slot=SlotType.ENTITIES,
        timeout_ms=12_345,
    )
    givenRegistry = _StubRegistry(
        {
            **_canonical_registry()._entries,
            "tabiya.ner.v1": _resolved(custom_ner, base_url=CORE_URL),
        }
    )
    givenPipeline = _canonical_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_happy_path(fake_http)
    executor = _executor(fake_http, registry=givenRegistry)

    # WHEN we run it
    await executor.run(
        pipeline=givenPipeline,
        source_overrides={"text": "job"},
        request_id="req-1",
        user_id="uid-1",
    )

    # THEN the NER call used a 12.345s timeout
    ner_call = fake_http.calls[1]
    assert ner_call["timeout"] == 12.345


async def test_run_raises_plugin_upstream_unavailable_on_503() -> None:
    # GIVEN a NEL that returns 503 with an UPSTREAM_UNAVAILABLE envelope
    givenPipeline = _canonical_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_happy_path(fake_http)
    fake_http.on(
        _plugin_url("tabiya.nel.v1"),
        httpx.Response(
            503,
            json={
                "code": "UPSTREAM_UNAVAILABLE",
                "message": "Embeddings cache not ready.",
            },
        ),
    )
    executor = _executor(fake_http)

    # WHEN we run
    # THEN we get PluginUpstreamUnavailableError with stage_index=2
    with pytest.raises(PluginUpstreamUnavailableError) as exc_info:
        await executor.run(
            pipeline=givenPipeline,
            source_overrides={"text": "job"},
            request_id="req-1",
            user_id="uid-1",
        )
    assert exc_info.value.stage_index == 2
    assert exc_info.value.plugin_id == "tabiya.nel.v1"


async def test_run_raises_plugin_invocation_error_on_500() -> None:
    # GIVEN a NER that returns 500
    givenPipeline = _canonical_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_happy_path(fake_http)
    fake_http.on(
        _plugin_url("tabiya.ner.v1"),
        httpx.Response(500, json={"code": "PLUGIN_INTERNAL", "message": "kaboom"}),
    )
    executor = _executor(fake_http)

    # WHEN we run
    # THEN PluginInvocationError with the stage index
    with pytest.raises(PluginInvocationError) as exc_info:
        await executor.run(
            pipeline=givenPipeline,
            source_overrides={"text": "job"},
            request_id="req-1",
            user_id="uid-1",
        )
    assert exc_info.value.stage_index == 1
    assert exc_info.value.plugin_id == "tabiya.ner.v1"
    assert exc_info.value.detail["http_status"] == 500


async def test_run_raises_plugin_timeout_on_httpx_timeout() -> None:
    # GIVEN a NEL that times out at the network level
    givenPipeline = _canonical_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_happy_path(fake_http)
    fake_http.raise_on(_plugin_url("tabiya.nel.v1"), httpx.ReadTimeout("slow"))
    executor = _executor(fake_http)

    # WHEN we run
    # THEN PluginTimeoutError
    with pytest.raises(PluginTimeoutError) as exc_info:
        await executor.run(
            pipeline=givenPipeline,
            source_overrides={"text": "job"},
            request_id="req-1",
            user_id="uid-1",
        )
    assert exc_info.value.stage_index == 2


async def test_run_raises_plugin_invocation_error_on_transport_error() -> None:
    # GIVEN a NER that's unreachable
    givenPipeline = _canonical_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_happy_path(fake_http)
    fake_http.raise_on(_plugin_url("tabiya.ner.v1"), httpx.ConnectError("refused"))
    executor = _executor(fake_http)

    # WHEN we run
    # THEN PluginInvocationError with UNAVAILABLE code
    with pytest.raises(PluginInvocationError) as exc_info:
        await executor.run(
            pipeline=givenPipeline,
            source_overrides={"text": "job"},
            request_id="req-1",
            user_id="uid-1",
        )
    assert exc_info.value.stage_index == 1


async def test_run_raises_when_stage_plugin_id_is_not_in_catalog() -> None:
    # GIVEN a pipeline whose second stage references a plugin the registry doesn't know
    givenPipeline = _canonical_pipeline()
    givenPipeline.stages[1] = StageDocument(plugin_id="tabiya.ghost.v1", config={})
    fake_http = _FakeAsyncClient()
    _wire_happy_path(fake_http)
    executor = _executor(fake_http)

    # WHEN we run
    # THEN PluginInvocationError before any network call for that stage
    with pytest.raises(PluginInvocationError) as exc_info:
        await executor.run(
            pipeline=givenPipeline,
            source_overrides={"text": "job"},
            request_id="req-1",
            user_id="uid-1",
        )
    assert exc_info.value.plugin_id == "tabiya.ghost.v1"
    # Only stage 0 (source) was invoked before the executor bailed.
    assert len(fake_http.calls) == 1


async def test_run_raises_when_plugin_is_coming_soon() -> None:
    # GIVEN a pipeline whose stage references a coming_soon plugin
    givenPipeline = _canonical_pipeline()
    givenPipeline.stages[1] = StageDocument(plugin_id="tabiya.source.scraper.v1", config={})
    givenRegistry = _StubRegistry(
        {
            **_canonical_registry()._entries,
            "tabiya.source.scraper.v1": ResolvedPlugin(
                plugin_id="tabiya.source.scraper.v1",
                coming_soon=True,
                status=PluginStatus.UNAVAILABLE,
            ),
        }
    )
    fake_http = _FakeAsyncClient()
    _wire_happy_path(fake_http)
    executor = _executor(fake_http, registry=givenRegistry)

    # WHEN we run
    # THEN PluginInvocationError with a coming-soon message
    with pytest.raises(PluginInvocationError) as exc_info:
        await executor.run(
            pipeline=givenPipeline,
            source_overrides={"text": "job"},
            request_id="req-1",
            user_id="uid-1",
        )
    assert "Coming Soon" in str(exc_info.value)


async def test_run_records_stage_metadata_from_response_body() -> None:
    # GIVEN the canonical happy path
    givenPipeline = _canonical_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_happy_path(fake_http)
    executor = _executor(fake_http)

    # WHEN we run
    result = await executor.run(
        pipeline=givenPipeline,
        source_overrides={"text": "job"},
        request_id="req-1",
        user_id="uid-1",
    )

    # THEN NER's outcome carries the per-stage metadata the plugin returned
    ner_outcome = result.stages[1]
    assert ner_outcome.metadata == {"model_name": "test-ner", "processing_time_ms": 2.0}

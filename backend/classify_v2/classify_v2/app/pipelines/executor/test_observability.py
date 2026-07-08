"""Observability tests for the executor.

Verifies design §11 log shape: one line per plugin /invoke, on the
`classify_v2.pipeline_stage` logger, with the pinned field set.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Optional

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
    PluginTimeoutError,
    PluginUpstreamUnavailableError,
)
from classify_v2.app.pipelines.registry import PluginStatus, ResolvedPlugin
from classify_v2.app.pipelines.repository import PipelineDocument, StageDocument


CORE_URL = "http://tabiya-core:5010"


class _FakeAsyncClient:
    def __init__(self) -> None:
        self._responses: dict[str, Callable[[dict], httpx.Response]] = {}
        self._exceptions: dict[str, Exception] = {}

    def on(self, url: str, response: httpx.Response) -> "_FakeAsyncClient":
        self._responses[url] = lambda body: response
        return self

    def raise_on(self, url: str, exc: Exception) -> "_FakeAsyncClient":
        self._exceptions[url] = exc
        return self

    async def post(self, url, *, json=None, headers=None, timeout=None):
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
    version: str = "0.1.0",
) -> Manifest:
    return Manifest(
        plugin_id=plugin_id,
        name=plugin_id,
        version=version,
        category=category,
        summary="test",
        icon="ner",
        input_slot=Slot(type=input_slot, cardinality="none" if input_slot == SlotType.NONE else "single"),
        output_slot=Slot(type=output_slot, cardinality="none" if output_slot == SlotType.NONE else "single"),
        config_schema={},
        timeout_ms=5_000,
        **{"x-tabiya-contract-version": CONTRACT_VERSION},
    )


def _resolved(manifest: Manifest) -> ResolvedPlugin:
    return ResolvedPlugin(
        plugin_id=manifest.plugin_id,
        resolved_url=f"{CORE_URL}/plugin/{manifest.plugin_id}",
        manifest=manifest,
        status=PluginStatus.ENABLED,
    )


class _StubRegistry:
    def __init__(self, entries: dict[str, ResolvedPlugin]) -> None:
        self._entries = entries

    def get(self, plugin_id: str) -> Optional[ResolvedPlugin]:
        return self._entries.get(plugin_id)


def _two_stage_pipeline() -> PipelineDocument:
    now = datetime.now(timezone.utc)
    return PipelineDocument(
        pipeline_id="pipe-obs-1",
        user_id="uid-1",
        name="Obs Test",
        stages=[
            StageDocument(plugin_id="tabiya.source.text.v1", config={"text": ""}),
            StageDocument(plugin_id="tabiya.ner.v1", config={}),
        ],
        is_active=False,
        is_default=False,
        is_readonly=False,
        created_at=now,
        updated_at=now,
    )


def _canonical_registry() -> _StubRegistry:
    return _StubRegistry(
        {
            "tabiya.source.text.v1": _resolved(
                _manifest(
                    "tabiya.source.text.v1",
                    category=PluginCategory.SOURCE,
                    input_slot=SlotType.NONE,
                    output_slot=SlotType.RAW_TEXT,
                    version="0.9.9",
                )
            ),
            "tabiya.ner.v1": _resolved(
                _manifest(
                    "tabiya.ner.v1",
                    category=PluginCategory.CORE,
                    input_slot=SlotType.RAW_TEXT,
                    output_slot=SlotType.ENTITIES,
                    version="1.2.3",
                )
            ),
        }
    )


def _wire_source_and_ner(http: _FakeAsyncClient) -> None:
    http.on(
        f"{CORE_URL}/plugin/tabiya.source.text.v1/invoke",
        httpx.Response(200, json={"output": {"text": "hi"}}),
    )
    http.on(
        f"{CORE_URL}/plugin/tabiya.ner.v1/invoke",
        httpx.Response(
            200,
            json={
                "output": {"entities": [], "source_text": "hi"},
                "metadata": {"model_name": "test-ner"},
            },
        ),
    )


def _stage_log_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records if record.name == "classify_v2.pipeline_stage"
    ]


async def test_emits_one_log_line_per_stage_on_happy_path(caplog) -> None:
    # GIVEN a two-stage pipeline where both stages return 200
    givenPipeline = _two_stage_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_source_and_ner(fake_http)
    executor = PipelineExecutor(
        registry=_canonical_registry(),  # type: ignore[arg-type]
        http_client=fake_http,  # type: ignore[arg-type]
    )

    # WHEN we run
    with caplog.at_level(logging.INFO, logger="classify_v2.pipeline_stage"):
        await executor.run(
            pipeline=givenPipeline,
            source_overrides={"text": "hi"},
            request_id="req-99",
            user_id="uid-1",
        )

    # THEN one log record per stage on the pipeline_stage logger
    stage_records = _stage_log_records(caplog)
    expectedRecordCount = 2
    assert len(stage_records) == expectedRecordCount


async def test_stage_log_record_carries_pinned_field_set(caplog) -> None:
    # GIVEN a happy-path run
    givenPipeline = _two_stage_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_source_and_ner(fake_http)
    executor = PipelineExecutor(
        registry=_canonical_registry(),  # type: ignore[arg-type]
        http_client=fake_http,  # type: ignore[arg-type]
    )

    # WHEN we run
    with caplog.at_level(logging.INFO, logger="classify_v2.pipeline_stage"):
        await executor.run(
            pipeline=givenPipeline,
            source_overrides={"text": "hi"},
            request_id="req-99",
            user_id="uid-1",
        )

    # THEN every design-§11 field is present on each record
    stage_records = _stage_log_records(caplog)
    expectedFields = {
        "pipeline_id",
        "stage_index",
        "plugin_id",
        "plugin_version",
        "category",
        "duration_ms",
        "status",
        "request_id",
    }
    for record in stage_records:
        for field in expectedFields:
            assert hasattr(record, field), f"missing {field} on stage log record"


async def test_stage_log_records_reflect_stage_ordering_and_plugin_versions(caplog) -> None:
    # GIVEN a two-stage pipeline
    givenPipeline = _two_stage_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_source_and_ner(fake_http)
    executor = PipelineExecutor(
        registry=_canonical_registry(),  # type: ignore[arg-type]
        http_client=fake_http,  # type: ignore[arg-type]
    )

    # WHEN we run
    with caplog.at_level(logging.INFO, logger="classify_v2.pipeline_stage"):
        await executor.run(
            pipeline=givenPipeline,
            source_overrides={"text": "hi"},
            request_id="req-42",
            user_id="uid-1",
        )

    # THEN stage_index goes 0 → 1 and plugin_version comes from the manifest
    stage_records = _stage_log_records(caplog)
    assert [record.stage_index for record in stage_records] == [0, 1]
    assert [record.plugin_id for record in stage_records] == [
        "tabiya.source.text.v1",
        "tabiya.ner.v1",
    ]
    assert [record.plugin_version for record in stage_records] == ["0.9.9", "1.2.3"]
    assert [record.category for record in stage_records] == ["source", "core"]
    assert all(record.pipeline_id == "pipe-obs-1" for record in stage_records)
    assert all(record.request_id == "req-42" for record in stage_records)
    assert all(record.status == "ok" for record in stage_records)


async def test_stage_log_carries_error_code_on_upstream_unavailable(caplog) -> None:
    # GIVEN a two-stage pipeline where stage 1 returns 503
    givenPipeline = _two_stage_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_source_and_ner(fake_http)
    fake_http.on(
        f"{CORE_URL}/plugin/tabiya.ner.v1/invoke",
        httpx.Response(
            503, json={"code": "UPSTREAM_UNAVAILABLE", "message": "cache cold"}
        ),
    )
    executor = PipelineExecutor(
        registry=_canonical_registry(),  # type: ignore[arg-type]
        http_client=fake_http,  # type: ignore[arg-type]
    )

    # WHEN we run
    with caplog.at_level(logging.INFO, logger="classify_v2.pipeline_stage"):
        with pytest.raises(PluginUpstreamUnavailableError):
            await executor.run(
                pipeline=givenPipeline,
                source_overrides={"text": "hi"},
                request_id="req-1",
                user_id="uid-1",
            )

    # THEN the failing stage's log carries status=error + error_code=UPSTREAM_UNAVAILABLE
    stage_records = _stage_log_records(caplog)
    expectedRecordCount = 2  # source ok + ner error
    assert len(stage_records) == expectedRecordCount
    failing = stage_records[-1]
    assert failing.status == "error"
    assert failing.error_code == "UPSTREAM_UNAVAILABLE"
    assert failing.plugin_id == "tabiya.ner.v1"


async def test_stage_log_carries_timeout_status_on_httpx_timeout(caplog) -> None:
    # GIVEN a stage that times out at the network level
    givenPipeline = _two_stage_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_source_and_ner(fake_http)
    fake_http.raise_on(
        f"{CORE_URL}/plugin/tabiya.ner.v1/invoke", httpx.ReadTimeout("slow")
    )
    executor = PipelineExecutor(
        registry=_canonical_registry(),  # type: ignore[arg-type]
        http_client=fake_http,  # type: ignore[arg-type]
    )

    # WHEN we run
    with caplog.at_level(logging.INFO, logger="classify_v2.pipeline_stage"):
        with pytest.raises(PluginTimeoutError):
            await executor.run(
                pipeline=givenPipeline,
                source_overrides={"text": "hi"},
                request_id="req-1",
                user_id="uid-1",
            )

    # THEN the failing stage logged status=timeout + error_code=TIMEOUT
    stage_records = _stage_log_records(caplog)
    failing = stage_records[-1]
    assert failing.status == "timeout"
    assert failing.error_code == "TIMEOUT"


async def test_stage_log_still_emits_when_plugin_is_not_in_catalog(caplog) -> None:
    # GIVEN a pipeline whose stage 1 references an unknown plugin
    givenPipeline = _two_stage_pipeline()
    givenPipeline.stages[1] = StageDocument(plugin_id="tabiya.ghost.v1", config={})
    fake_http = _FakeAsyncClient()
    _wire_source_and_ner(fake_http)
    executor = PipelineExecutor(
        registry=_canonical_registry(),  # type: ignore[arg-type]
        http_client=fake_http,  # type: ignore[arg-type]
    )

    # WHEN we run
    with caplog.at_level(logging.INFO, logger="classify_v2.pipeline_stage"):
        with pytest.raises(Exception):
            await executor.run(
                pipeline=givenPipeline,
                source_overrides={"text": "hi"},
                request_id="req-1",
                user_id="uid-1",
            )

    # THEN the failing stage still emitted a log line, marked error/UNAVAILABLE
    stage_records = _stage_log_records(caplog)
    failing = [record for record in stage_records if record.plugin_id == "tabiya.ghost.v1"]
    assert failing, "expected a log record for the unresolved plugin"
    assert failing[0].status == "error"
    assert failing[0].error_code == "UNAVAILABLE"


async def test_stage_log_reports_duration_ms(caplog) -> None:
    # GIVEN a happy-path run
    givenPipeline = _two_stage_pipeline()
    fake_http = _FakeAsyncClient()
    _wire_source_and_ner(fake_http)
    executor = PipelineExecutor(
        registry=_canonical_registry(),  # type: ignore[arg-type]
        http_client=fake_http,  # type: ignore[arg-type]
    )

    # WHEN we run
    with caplog.at_level(logging.INFO, logger="classify_v2.pipeline_stage"):
        await executor.run(
            pipeline=givenPipeline,
            source_overrides={"text": "hi"},
            request_id="req-1",
            user_id="uid-1",
        )

    # THEN every stage log has a non-negative duration_ms
    for record in _stage_log_records(caplog):
        assert isinstance(record.duration_ms, (int, float))
        assert record.duration_ms >= 0

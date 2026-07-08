"""Tests for the plugin-based ClassifyService.

The service is now a thin adapter over `PipelineExecutor`. These tests
inject a stub executor to verify the mapping from `ExecutorResult` back
into `ClassifyResponse` — the executor itself is covered separately.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from classify_v2.app.classification.service.errors import (
    EmbeddingsCacheNotReadyError,
    NELServiceError,
    NERServiceError,
)
from classify_v2.app.classification.service.service import ClassifyService
from classify_v2.app.classification.service.types import ClassifyOptions
from classify_v2.app.pipelines.executor import (
    ExecutorResult,
    PluginInvocationError,
    PluginTimeoutError,
    PluginUpstreamUnavailableError,
    StageOutcome,
)
from classify_v2.app.pipelines.executor.executor import PipelineExecutor
from classify_v2.app.pipelines.repository import PipelineDocument, StageDocument


def _canonical_pipeline_doc() -> PipelineDocument:
    now = datetime.now(timezone.utc)
    return PipelineDocument(
        pipeline_id="pipe-1",
        user_id="uid-1",
        name="Default Tabiya",
        stages=[
            StageDocument(plugin_id="tabiya.source.text.v1", config={"text": ""}),
            StageDocument(plugin_id="tabiya.ner.v1", config={}),
            StageDocument(
                plugin_id="tabiya.nel.v1",
                config={"nel_model_id": "m", "taxonomy_model_id": "t"},
            ),
            StageDocument(plugin_id="tabiya.sink.results.v1", config={}),
        ],
        is_active=True,
        is_default=True,
        is_readonly=True,
        created_at=now,
        updated_at=now,
    )


def _canonical_executor_result(*, source_text: str = "Statistician") -> ExecutorResult:
    return ExecutorResult(
        pipeline_id="pipe-1",
        pipeline_name="Default Tabiya",
        stages=[
            StageOutcome(
                stage_index=0,
                plugin_id="tabiya.source.text.v1",
                plugin_version="0.1.0",
                category="source",
                duration_ms=1.0,
                status="ok",
            ),
            StageOutcome(
                stage_index=1,
                plugin_id="tabiya.ner.v1",
                plugin_version="0.1.0",
                category="core",
                duration_ms=2.0,
                status="ok",
                metadata={"model_name": "ner-test", "processing_time_ms": 2.0},
            ),
            StageOutcome(
                stage_index=2,
                plugin_id="tabiya.nel.v1",
                plugin_version="0.1.0",
                category="core",
                duration_ms=3.0,
                status="ok",
                metadata={"nel_model_id": "all-MiniLM-L6-v2", "taxonomy_model_id": "tax-1"},
            ),
            StageOutcome(
                stage_index=3,
                plugin_id="tabiya.sink.results.v1",
                plugin_version="0.1.0",
                category="sink",
                duration_ms=0.5,
                status="ok",
            ),
        ],
        final_output={"kind": "None"},
        linked_entities_payload={
            "entities": [
                {
                    "surface_form": source_text,
                    "entity_type": "occupation",
                    "span": {"start": 0, "end": len(source_text)},
                    "matches": [
                        {
                            "id": f"esco/occupation/{source_text}",
                            "preferred_label": source_text,
                            "score": 0.91,
                            "uri": f"http://taxonomy.tabiya.tech/occupation/{source_text}",
                        }
                    ],
                }
            ],
            "source_text": source_text,
        },
    )


class _StubExecutor:
    def __init__(self, *, result: ExecutorResult | None = None, exc: Exception | None = None) -> None:
        self._result = result
        self._exc = exc
        self.last_call: dict = {}

    async def run(self, **kwargs) -> ExecutorResult:
        self.last_call = kwargs
        if self._exc is not None:
            raise self._exc
        assert self._result is not None
        return self._result


def _service_with_stub_executor(stub: _StubExecutor) -> ClassifyService:
    # The concrete PipelineExecutor takes registry + http_client, but the
    # service treats it as an "async .run(...)" collaborator — we can
    # substitute a plain stub without instantiating the real executor.
    service = ClassifyService(executor=stub)  # type: ignore[arg-type]
    return service


async def test_classify_maps_linked_entities_to_classify_response() -> None:
    # GIVEN a canonical executor result
    givenPipeline = _canonical_pipeline_doc()
    givenResult = _canonical_executor_result(source_text="Statistician")
    stub = _StubExecutor(result=givenResult)
    service = _service_with_stub_executor(stub)

    # WHEN we classify
    response = await service.classify(
        pipeline=givenPipeline,
        input_text="Statistician wanted",
        options=ClassifyOptions(),
        request_id="req-1",
        user_id="uid-1",
    )

    # THEN entities carry surface_form + matches from the linked payload
    assert len(response.entities) == 1
    assert response.entities[0].surface_form == "Statistician"
    assert response.entities[0].matches[0].similarity_score == 0.91


async def test_classify_populates_metadata_from_stage_outcomes() -> None:
    # GIVEN a canonical executor result
    givenPipeline = _canonical_pipeline_doc()
    givenResult = _canonical_executor_result()
    stub = _StubExecutor(result=givenResult)
    service = _service_with_stub_executor(stub)

    # WHEN we classify
    response = await service.classify(
        pipeline=givenPipeline,
        input_text="Statistician",
        options=ClassifyOptions(),
        request_id="req-1",
        user_id="uid-1",
    )

    # THEN ner_model + nel_model_id + taxonomy_model_id come from per-stage metadata
    assert response.metadata.ner_model == "ner-test"
    assert response.metadata.nel_model_id == "all-MiniLM-L6-v2"
    assert response.metadata.taxonomy_model_id == "tax-1"


async def test_classify_response_includes_pipeline_summary() -> None:
    # GIVEN a canonical executor result
    givenPipeline = _canonical_pipeline_doc()
    givenResult = _canonical_executor_result()
    stub = _StubExecutor(result=givenResult)
    service = _service_with_stub_executor(stub)

    # WHEN we classify
    response = await service.classify(
        pipeline=givenPipeline,
        input_text="Statistician",
        options=ClassifyOptions(),
        request_id="req-1",
        user_id="uid-1",
    )

    # THEN metadata.pipeline lists every stage
    expectedIds = [
        "tabiya.source.text.v1",
        "tabiya.ner.v1",
        "tabiya.nel.v1",
        "tabiya.sink.results.v1",
    ]
    assert response.metadata.pipeline is not None
    assert response.metadata.pipeline.pipeline_id == "pipe-1"
    assert response.metadata.pipeline.name == "Default Tabiya"
    assert [stage.plugin_id for stage in response.metadata.pipeline.stages] == expectedIds
    assert [stage.category for stage in response.metadata.pipeline.stages] == [
        "source",
        "core",
        "core",
        "sink",
    ]


async def test_classify_passes_source_override_with_text_key_to_executor() -> None:
    # GIVEN a stub executor and a canonical pipeline
    givenPipeline = _canonical_pipeline_doc()
    stub = _StubExecutor(result=_canonical_executor_result())
    service = _service_with_stub_executor(stub)
    givenText = "New job ad text"

    # WHEN we classify
    await service.classify(
        pipeline=givenPipeline,
        input_text=givenText,
        options=ClassifyOptions(),
        request_id="req-1",
        user_id="uid-1",
    )

    # THEN the executor was called with source_overrides = {"text": givenText}
    assert stub.last_call["source_overrides"] == {"text": givenText}


async def test_classify_wraps_upstream_unavailable_as_embeddings_cache_not_ready() -> None:
    # GIVEN an executor that raises PluginUpstreamUnavailableError
    givenPipeline = _canonical_pipeline_doc()
    stub = _StubExecutor(
        exc=PluginUpstreamUnavailableError(
            "Embeddings cache not ready.",
            stage_index=2,
            plugin_id="tabiya.nel.v1",
        )
    )
    service = _service_with_stub_executor(stub)

    # WHEN we classify
    # THEN we get EmbeddingsCacheNotReadyError so the route emits 503
    with pytest.raises(EmbeddingsCacheNotReadyError):
        await service.classify(
            pipeline=givenPipeline,
            input_text="Statistician",
            options=ClassifyOptions(),
            request_id="req-1",
            user_id="uid-1",
        )


async def test_classify_wraps_plugin_timeout_as_nel_service_error() -> None:
    # GIVEN an executor that raises PluginTimeoutError
    givenPipeline = _canonical_pipeline_doc()
    stub = _StubExecutor(
        exc=PluginTimeoutError(
            "Plugin timed out.",
            stage_index=1,
            plugin_id="tabiya.ner.v1",
        )
    )
    service = _service_with_stub_executor(stub)

    # WHEN we classify
    # THEN NELServiceError so the route emits 504
    with pytest.raises(NELServiceError):
        await service.classify(
            pipeline=givenPipeline,
            input_text="Statistician",
            options=ClassifyOptions(),
            request_id="req-1",
            user_id="uid-1",
        )


async def test_classify_wraps_plugin_invocation_error_as_ner_service_error() -> None:
    # GIVEN an executor that raises PluginInvocationError
    givenPipeline = _canonical_pipeline_doc()
    stub = _StubExecutor(
        exc=PluginInvocationError(
            "Plugin returned 500.",
            stage_index=1,
            plugin_id="tabiya.ner.v1",
        )
    )
    service = _service_with_stub_executor(stub)

    # WHEN we classify
    # THEN NERServiceError so the route emits 502
    with pytest.raises(NERServiceError):
        await service.classify(
            pipeline=givenPipeline,
            input_text="Statistician",
            options=ClassifyOptions(),
            request_id="req-1",
            user_id="uid-1",
        )


async def test_classify_returns_empty_entities_when_pipeline_had_no_nel_stage() -> None:
    # GIVEN an executor result without a LinkedEntities payload (e.g. a
    # source-only pipeline that a future test might build)
    givenPipeline = _canonical_pipeline_doc()
    givenResult = _canonical_executor_result()
    givenResult.linked_entities_payload = None
    stub = _StubExecutor(result=givenResult)
    service = _service_with_stub_executor(stub)

    # WHEN we classify
    response = await service.classify(
        pipeline=givenPipeline,
        input_text="Statistician",
        options=ClassifyOptions(),
        request_id="req-1",
        user_id="uid-1",
    )

    # THEN entities is empty rather than crashing
    assert response.entities == []

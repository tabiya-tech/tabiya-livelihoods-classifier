"""Tests for classify v2 routes.

Overrides both the pipeline service (source of the resolved pipeline) and
the classify service (the executor-backed core) via FastAPI dependency
overrides, so the tests never touch Mongo or the plugin registry.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from classify_v2.app.classification.routes.routes import (
    _get_classify_service,
    _get_pipeline_service,
    router,
)
from classify_v2.app.classification.service.errors import (
    EmbeddingsCacheNotReadyError,
    NELServiceError,
    NERServiceError,
)
from classify_v2.app.classification.service.service import IClassifyService
from classify_v2.app.classification.service.types import (
    ClassifiedEntity,
    ClassifyMetadata,
    ClassifyResponse,
    EntitySpan,
    OccupationEntity,
    OccupationMatch,
    PipelineStageSummary,
    PipelineSummary,
)
from classify_v2.app.pipelines.repository import (
    PipelineDocument,
    PipelineNotFoundError,
    StageDocument,
)
from classify_v2.app.pipelines.service import IPipelineService
from classify_v2.app.pipelines.service.service import DefaultTabiyaConfig


@pytest.fixture(autouse=True)
def _force_local_mode(monkeypatch):
    monkeypatch.setenv("TARGET_ENVIRONMENT_TYPE", "local")


def _pipeline_doc(*, pipeline_id: str = "pipe-1", is_active: bool = True) -> PipelineDocument:
    now = datetime.now(timezone.utc)
    return PipelineDocument(
        pipeline_id=pipeline_id,
        user_id="local-user",
        name="Test Pipeline",
        stages=[
            StageDocument(plugin_id="tabiya.source.text.v1", config={"text": ""}),
            StageDocument(plugin_id="tabiya.sink.results.v1", config={}),
        ],
        is_active=is_active,
        is_default=False,
        is_readonly=False,
        created_at=now,
        updated_at=now,
    )


class _StubPipelineService(IPipelineService):
    def __init__(self, docs: list[PipelineDocument]) -> None:
        self._docs = docs
        self.ensure_default_called = False
        self.get_calls: list[str] = []

    async def list_for_user(self, user_id: str) -> list[PipelineDocument]:
        return [doc for doc in self._docs if doc.user_id == user_id]

    async def get(self, *, user_id: str, pipeline_id: str) -> PipelineDocument:
        self.get_calls.append(pipeline_id)
        for doc in self._docs:
            if doc.user_id == user_id and doc.pipeline_id == pipeline_id:
                return doc
        raise PipelineNotFoundError(pipeline_id, user_id)

    async def create(self, **kwargs):  # noqa: D401 — unused in these tests
        raise NotImplementedError

    async def update(self, **kwargs):
        raise NotImplementedError

    async def delete(self, **kwargs) -> None:
        raise NotImplementedError

    async def activate(self, **kwargs):
        raise NotImplementedError

    async def clone(self, **kwargs):
        raise NotImplementedError

    def validate(self, stages):
        return []

    async def ensure_default(
        self, *, user_id: str, default_config: DefaultTabiyaConfig
    ) -> PipelineDocument:
        self.ensure_default_called = True
        seeded = _pipeline_doc(pipeline_id="default-id")
        self._docs.append(seeded)
        return seeded


class _StubClassifyService(IClassifyService):
    def __init__(
        self,
        *,
        response: Optional[ClassifyResponse] = None,
        raises: Optional[Exception] = None,
    ) -> None:
        self._response = response
        self._raises = raises
        self.last_call: dict[str, Any] = {}

    async def classify(self, **kwargs) -> ClassifyResponse:
        self.last_call = kwargs
        if self._raises is not None:
            raise self._raises
        assert self._response is not None
        return self._response


def _response_with_summary(pipeline_id: str = "pipe-1") -> ClassifyResponse:
    return ClassifyResponse(
        entities=[
            ClassifiedEntity(
                entity_type="occupation",
                surface_form="Head Chef",
                span=EntitySpan(start=0, end=9),
                matches=[
                    OccupationMatch(
                        similarity_score=0.9,
                        entity=OccupationEntity(
                            uuid="u1",
                            origin_uuid="u1",
                            uuid_history=["u1"],
                            preferred_label="Head Chef",
                            origin_uri="http://example.com",
                            alt_labels=[],
                            description="",
                        ),
                    )
                ],
            )
        ],
        metadata=ClassifyMetadata(
            classifier_version="2.0.0",
            ner_model="ner-model",
            nel_model_id="nel-1",
            taxonomy_model_id="tax-1",
            processing_time_ms=100.0,
            pipeline=PipelineSummary(
                pipeline_id=pipeline_id,
                name="Test Pipeline",
                stages=[
                    PipelineStageSummary(plugin_id="tabiya.source.text.v1", category="source"),
                    PipelineStageSummary(plugin_id="tabiya.sink.results.v1", category="sink"),
                ],
            ),
        ),
    )


def _build_app(
    pipeline_service: IPipelineService, classify_service: IClassifyService
) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[_get_pipeline_service] = lambda: pipeline_service
    app.dependency_overrides[_get_classify_service] = lambda: classify_service
    return app


async def _post(app: FastAPI, body: dict[str, Any]):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/v2/classify", json=body)


async def test_classify_uses_users_active_pipeline_by_default() -> None:
    # GIVEN a user with one active pipeline
    givenPipeline = _pipeline_doc(pipeline_id="active-1", is_active=True)
    pipeline_svc = _StubPipelineService([givenPipeline])
    classify_svc = _StubClassifyService(response=_response_with_summary("active-1"))
    app = _build_app(pipeline_svc, classify_svc)

    # WHEN we POST /v2/classify
    response = await _post(app, {"text": "Head Chef needed"})

    # THEN the classify service was called with the active pipeline
    expectedStatus = 200
    assert response.status_code == expectedStatus
    assert classify_svc.last_call["pipeline"].pipeline_id == "active-1"


async def test_classify_uses_explicit_pipeline_id_when_provided() -> None:
    # GIVEN a user with two pipelines, one active
    docs = [
        _pipeline_doc(pipeline_id="active-1", is_active=True),
        _pipeline_doc(pipeline_id="alt-2", is_active=False),
    ]
    pipeline_svc = _StubPipelineService(docs)
    classify_svc = _StubClassifyService(response=_response_with_summary("alt-2"))
    app = _build_app(pipeline_svc, classify_svc)

    # WHEN we POST with an explicit pipeline_id
    response = await _post(
        app, {"text": "Head Chef needed", "pipeline_id": "alt-2"}
    )

    # THEN classify used the override, not the active pipeline
    expectedStatus = 200
    assert response.status_code == expectedStatus
    assert classify_svc.last_call["pipeline"].pipeline_id == "alt-2"


async def test_classify_returns_404_when_explicit_pipeline_missing() -> None:
    # GIVEN a user with no pipeline matching the requested id
    pipeline_svc = _StubPipelineService([])
    classify_svc = _StubClassifyService(response=_response_with_summary())
    app = _build_app(pipeline_svc, classify_svc)

    # WHEN we POST with an unknown pipeline_id
    response = await _post(
        app, {"text": "Head Chef needed", "pipeline_id": "ghost"}
    )

    # THEN 404
    expectedStatus = 404
    assert response.status_code == expectedStatus


async def test_classify_seeds_default_when_no_active_pipeline_and_env_configured(
    monkeypatch,
) -> None:
    # GIVEN a user with no pipelines and seeding env vars set
    monkeypatch.setenv("DEFAULT_NEL_MODEL_ID", "all-MiniLM-L6-v2")
    monkeypatch.setenv("DEFAULT_TAXONOMY_MODEL_ID", "model-abc")
    pipeline_svc = _StubPipelineService([])
    classify_svc = _StubClassifyService(response=_response_with_summary("default-id"))
    app = _build_app(pipeline_svc, classify_svc)

    # WHEN we POST
    response = await _post(app, {"text": "Head Chef needed"})

    # THEN a Default Tabiya was seeded and classify ran against it
    expectedStatus = 200
    assert response.status_code == expectedStatus
    assert pipeline_svc.ensure_default_called is True
    assert classify_svc.last_call["pipeline"].pipeline_id == "default-id"


async def test_classify_returns_400_when_no_pipeline_and_no_seeding_env(
    monkeypatch,
) -> None:
    # GIVEN a user with no pipelines and seeding env vars unset
    monkeypatch.delenv("DEFAULT_NEL_MODEL_ID", raising=False)
    monkeypatch.delenv("DEFAULT_TAXONOMY_MODEL_ID", raising=False)
    pipeline_svc = _StubPipelineService([])
    classify_svc = _StubClassifyService(response=_response_with_summary())
    app = _build_app(pipeline_svc, classify_svc)

    # WHEN we POST
    response = await _post(app, {"text": "Head Chef needed"})

    # THEN 400 with a helpful message pointing at /v2/pipelines
    expectedStatus = 400
    assert response.status_code == expectedStatus
    assert "/v2/pipelines" in response.json()["detail"]


async def test_classify_returns_400_when_no_text_or_title_description() -> None:
    # GIVEN a user with an active pipeline
    docs = [_pipeline_doc(is_active=True)]
    pipeline_svc = _StubPipelineService(docs)
    classify_svc = _StubClassifyService(response=_response_with_summary())
    app = _build_app(pipeline_svc, classify_svc)

    # WHEN we POST with neither text nor title+description
    response = await _post(app, {})

    # THEN 400
    expectedStatus = 400
    assert response.status_code == expectedStatus


async def test_classify_accepts_title_and_description() -> None:
    # GIVEN a user with an active pipeline
    docs = [_pipeline_doc(is_active=True)]
    pipeline_svc = _StubPipelineService(docs)
    classify_svc = _StubClassifyService(response=_response_with_summary())
    app = _build_app(pipeline_svc, classify_svc)

    # WHEN we POST title+description
    response = await _post(
        app, {"title": "Head Chef", "description": "Kitchen manager"}
    )

    # THEN 200 and the joined text was forwarded
    expectedStatus = 200
    assert response.status_code == expectedStatus
    assert "Head Chef" in classify_svc.last_call["input_text"]
    assert "Kitchen manager" in classify_svc.last_call["input_text"]


async def test_classify_returns_503_on_embeddings_cache_not_ready() -> None:
    # GIVEN the classify service raises EmbeddingsCacheNotReadyError
    docs = [_pipeline_doc(is_active=True)]
    pipeline_svc = _StubPipelineService(docs)
    classify_svc = _StubClassifyService(raises=EmbeddingsCacheNotReadyError("cache warming"))
    app = _build_app(pipeline_svc, classify_svc)

    # WHEN we POST
    response = await _post(app, {"text": "job"})

    # THEN 503
    expectedStatus = 503
    assert response.status_code == expectedStatus


async def test_classify_returns_504_on_plugin_timeout() -> None:
    # GIVEN the classify service raises NELServiceError (executor wrapper for timeouts)
    docs = [_pipeline_doc(is_active=True)]
    pipeline_svc = _StubPipelineService(docs)
    classify_svc = _StubClassifyService(raises=NELServiceError("timeout at stage 2"))
    app = _build_app(pipeline_svc, classify_svc)

    # WHEN we POST
    response = await _post(app, {"text": "job"})

    # THEN 504
    expectedStatus = 504
    assert response.status_code == expectedStatus


async def test_classify_returns_502_on_plugin_invocation_error() -> None:
    # GIVEN the classify service raises NERServiceError (executor wrapper for invocation failures)
    docs = [_pipeline_doc(is_active=True)]
    pipeline_svc = _StubPipelineService(docs)
    classify_svc = _StubClassifyService(raises=NERServiceError("plugin 500"))
    app = _build_app(pipeline_svc, classify_svc)

    # WHEN we POST
    response = await _post(app, {"text": "job"})

    # THEN 502
    expectedStatus = 502
    assert response.status_code == expectedStatus


async def test_classify_response_includes_pipeline_summary_in_metadata() -> None:
    # GIVEN an active pipeline and a canonical response
    docs = [_pipeline_doc(pipeline_id="p-1", is_active=True)]
    pipeline_svc = _StubPipelineService(docs)
    classify_svc = _StubClassifyService(response=_response_with_summary("p-1"))
    app = _build_app(pipeline_svc, classify_svc)

    # WHEN we POST
    response = await _post(app, {"text": "job"})

    # THEN the response body carries metadata.pipeline
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["pipeline"]["pipeline_id"] == "p-1"
    assert body["metadata"]["pipeline"]["stages"][0]["plugin_id"] == "tabiya.source.text.v1"


async def test_classify_emits_summary_log_line_after_a_successful_run(caplog) -> None:
    # GIVEN an active pipeline and a canonical response
    import logging as _logging

    docs = [_pipeline_doc(pipeline_id="p-summary", is_active=True)]
    pipeline_svc = _StubPipelineService(docs)
    classify_svc = _StubClassifyService(response=_response_with_summary("p-summary"))
    app = _build_app(pipeline_svc, classify_svc)

    # WHEN we POST and capture logs
    with caplog.at_level(_logging.INFO, logger="classify_v2.classify_summary"):
        response = await _post(app, {"text": "Head Chef needed"})

    # THEN one summary log record was emitted with the pinned fields
    assert response.status_code == 200
    summary_records = [
        record
        for record in caplog.records
        if record.name == "classify_v2.classify_summary"
    ]
    expectedSummaryCount = 1
    assert len(summary_records) == expectedSummaryCount
    record = summary_records[0]
    assert record.pipeline_id == "p-summary"
    assert record.pipeline_name == "Test Pipeline"
    assert record.entity_count == 1
    assert isinstance(record.total_duration_ms, (int, float))
    assert record.total_stages == 2
    assert record.request_id  # a uuid was minted by the route


async def test_classify_does_not_emit_summary_log_on_error(caplog) -> None:
    # GIVEN the classify service raises
    import logging as _logging

    docs = [_pipeline_doc(is_active=True)]
    pipeline_svc = _StubPipelineService(docs)
    classify_svc = _StubClassifyService(raises=NERServiceError("boom"))
    app = _build_app(pipeline_svc, classify_svc)

    # WHEN we POST
    with caplog.at_level(_logging.INFO, logger="classify_v2.classify_summary"):
        response = await _post(app, {"text": "job"})

    # THEN no summary log record was emitted (we only summarise successful runs)
    assert response.status_code == 502
    summary_records = [
        record
        for record in caplog.records
        if record.name == "classify_v2.classify_summary"
    ]
    assert summary_records == []

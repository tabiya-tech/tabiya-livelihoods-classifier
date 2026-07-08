"""Tests for /v2/pipelines CRUD routes.

Uses FastAPI dependency overrides to inject a stub service that captures
the call arguments — the routes are a thin translation layer, so the
tests verify that translation without needing Mongo.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from classify_v2.app.pipelines.repository import (
    PipelineDocument,
    PipelineNotFoundError,
    ReadonlyPipelineError,
    StageDocument,
)
from classify_v2.app.pipelines.routes.routes import (
    get_pipeline_service,
    router as pipelines_router,
)
from classify_v2.app.pipelines.service import (
    CreatePipelineInput,
    IPipelineService,
    PipelineValidationError,
    UpdatePipelineInput,
    ValidationIssue,
)
from classify_v2.app.pipelines.service.errors import IssueCode
from classify_v2.app.pipelines.service.service import DefaultTabiyaConfig


class _StubService(IPipelineService):
    def __init__(self) -> None:
        self._docs: list[PipelineDocument] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.raise_on: dict[str, Exception] = {}

    async def list_for_user(self, user_id: str) -> list[PipelineDocument]:
        self.calls.append(("list_for_user", {"user_id": user_id}))
        return [doc for doc in self._docs if doc.user_id == user_id]

    async def get(self, *, user_id: str, pipeline_id: str) -> PipelineDocument:
        self.calls.append(("get", {"user_id": user_id, "pipeline_id": pipeline_id}))
        if "get" in self.raise_on:
            raise self.raise_on["get"]
        for doc in self._docs:
            if doc.user_id == user_id and doc.pipeline_id == pipeline_id:
                return doc
        raise PipelineNotFoundError(pipeline_id, user_id)

    async def create(self, *, user_id: str, request: CreatePipelineInput) -> PipelineDocument:
        self.calls.append(("create", {"user_id": user_id, "request": request.model_dump()}))
        if "create" in self.raise_on:
            raise self.raise_on["create"]
        now = datetime.now(timezone.utc)
        doc = PipelineDocument(
            pipeline_id="new-id",
            user_id=user_id,
            name=request.name,
            stages=list(request.stages),
            is_active=False,
            is_default=False,
            is_readonly=False,
            created_at=now,
            updated_at=now,
        )
        self._docs.append(doc)
        return doc

    async def update(
        self, *, user_id: str, pipeline_id: str, request: UpdatePipelineInput
    ) -> PipelineDocument:
        self.calls.append(
            (
                "update",
                {
                    "user_id": user_id,
                    "pipeline_id": pipeline_id,
                    "request": request.model_dump(),
                },
            )
        )
        if "update" in self.raise_on:
            raise self.raise_on["update"]
        for index, doc in enumerate(self._docs):
            if doc.user_id == user_id and doc.pipeline_id == pipeline_id:
                updated = doc.model_copy(
                    update={"name": request.name, "stages": list(request.stages)}
                )
                self._docs[index] = updated
                return updated
        raise PipelineNotFoundError(pipeline_id, user_id)

    async def delete(self, *, user_id: str, pipeline_id: str) -> None:
        self.calls.append(("delete", {"user_id": user_id, "pipeline_id": pipeline_id}))
        if "delete" in self.raise_on:
            raise self.raise_on["delete"]
        self._docs = [
            doc for doc in self._docs
            if not (doc.user_id == user_id and doc.pipeline_id == pipeline_id)
        ]

    async def activate(self, *, user_id: str, pipeline_id: str) -> PipelineDocument:
        self.calls.append(("activate", {"user_id": user_id, "pipeline_id": pipeline_id}))
        if "activate" in self.raise_on:
            raise self.raise_on["activate"]
        for index, doc in enumerate(self._docs):
            if doc.user_id == user_id and doc.pipeline_id == pipeline_id:
                self._docs[index] = doc.model_copy(update={"is_active": True})
                return self._docs[index]
        raise PipelineNotFoundError(pipeline_id, user_id)

    async def clone(self, *, user_id: str, pipeline_id: str) -> PipelineDocument:
        self.calls.append(("clone", {"user_id": user_id, "pipeline_id": pipeline_id}))
        if "clone" in self.raise_on:
            raise self.raise_on["clone"]
        for doc in self._docs:
            if doc.user_id == user_id and doc.pipeline_id == pipeline_id:
                clone_doc = doc.model_copy(
                    update={
                        "pipeline_id": "clone-id",
                        "name": f"{doc.name} (copy)",
                        "is_active": False,
                        "is_default": False,
                        "is_readonly": False,
                    }
                )
                self._docs.append(clone_doc)
                return clone_doc
        raise PipelineNotFoundError(pipeline_id, user_id)

    def validate(self, stages: list[StageDocument]) -> list[ValidationIssue]:
        self.calls.append(("validate", {"stages": [stage.model_dump() for stage in stages]}))
        if "validate" in self.raise_on:
            raise self.raise_on["validate"]
        return []

    async def ensure_default(
        self, *, user_id: str, default_config: DefaultTabiyaConfig
    ) -> PipelineDocument:
        self.calls.append(("ensure_default", {"user_id": user_id}))
        now = datetime.now(timezone.utc)
        seed = PipelineDocument(
            pipeline_id="default-id",
            user_id=user_id,
            name="Default Tabiya",
            stages=[
                StageDocument(plugin_id="tabiya.source.text.v1", config={"text": ""}),
                StageDocument(plugin_id="tabiya.sink.results.v1", config={}),
            ],
            is_active=True,
            is_default=True,
            is_readonly=True,
            created_at=now,
            updated_at=now,
        )
        self._docs.append(seed)
        return seed


@pytest.fixture(autouse=True)
def _force_local_mode(monkeypatch):
    monkeypatch.setenv("TARGET_ENVIRONMENT_TYPE", "local")


@pytest.fixture
def client_and_service():
    app = FastAPI()
    app.include_router(pipelines_router)
    stub_service = _StubService()

    async def _override_service() -> IPipelineService:
        return stub_service

    app.dependency_overrides[get_pipeline_service] = _override_service
    return TestClient(app), stub_service


def _valid_stages_payload() -> list[dict]:
    return [
        {"plugin_id": "tabiya.source.text.v1", "config": {"text": ""}},
        {"plugin_id": "tabiya.sink.results.v1", "config": {}},
    ]


def test_get_list_returns_pipelines(client_and_service, monkeypatch) -> None:
    # GIVEN a stub service with one row AND seeding disabled
    monkeypatch.delenv("DEFAULT_NEL_MODEL_ID", raising=False)
    monkeypatch.delenv("DEFAULT_TAXONOMY_MODEL_ID", raising=False)
    client, service = client_and_service
    service._docs = [
        PipelineDocument(
            pipeline_id="p1",
            user_id="local-user",
            name="A",
            stages=[StageDocument(plugin_id="tabiya.source.text.v1", config={})],
            is_active=False,
            is_default=False,
            is_readonly=False,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
    ]

    # WHEN we hit list
    response = client.get("/v2/pipelines")

    # THEN 200 with the row
    expectedStatus = 200
    assert response.status_code == expectedStatus
    body = response.json()
    assert [pipeline["pipeline_id"] for pipeline in body["pipelines"]] == ["p1"]


def test_get_list_lazily_calls_ensure_default(client_and_service, monkeypatch) -> None:
    # GIVEN a caller with the seeding env vars set
    monkeypatch.setenv("DEFAULT_NEL_MODEL_ID", "all-MiniLM-L6-v2")
    monkeypatch.setenv("DEFAULT_TAXONOMY_MODEL_ID", "model-abc")
    client, service = client_and_service

    # WHEN we hit list
    response = client.get("/v2/pipelines")

    # THEN the service saw ensure_default
    expectedStatus = 200
    assert response.status_code == expectedStatus
    call_names = [name for name, _ in service.calls]
    assert "ensure_default" in call_names


def test_get_list_skips_ensure_default_when_seeding_env_unset(
    client_and_service, monkeypatch
) -> None:
    # GIVEN the seeding env vars are unset
    monkeypatch.delenv("DEFAULT_NEL_MODEL_ID", raising=False)
    monkeypatch.delenv("DEFAULT_TAXONOMY_MODEL_ID", raising=False)
    client, service = client_and_service

    # WHEN we hit list
    response = client.get("/v2/pipelines")

    # THEN the service was not asked to seed
    assert response.status_code == 200
    call_names = [name for name, _ in service.calls]
    assert "ensure_default" not in call_names


def test_create_persists_and_returns_the_pipeline(client_and_service) -> None:
    # GIVEN a valid payload
    client, service = client_and_service
    givenPayload = {"name": "My pipeline", "stages": _valid_stages_payload()}

    # WHEN we POST
    response = client.post("/v2/pipelines", json=givenPayload)

    # THEN 201 with the created row
    expectedStatus = 201
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["name"] == "My pipeline"
    assert body["pipeline_id"] == "new-id"


def test_create_returns_422_on_validation_error(client_and_service) -> None:
    # GIVEN the stub raises a validation error
    client, service = client_and_service
    givenIssues = [
        ValidationIssue(
            code=IssueCode.UNKNOWN_PLUGIN,
            message="Plugin 'tabiya.ghost.v1' is not in the catalog.",
            stage_index=0,
            plugin_id="tabiya.ghost.v1",
        )
    ]
    service.raise_on["create"] = PipelineValidationError(givenIssues)

    # WHEN we create
    response = client.post(
        "/v2/pipelines",
        json={"name": "Bad", "stages": _valid_stages_payload()},
    )

    # THEN 422 with the issues in the detail body
    expectedStatus = 422
    assert response.status_code == expectedStatus
    detail = response.json()["detail"]
    assert detail["issues"][0]["code"] == IssueCode.UNKNOWN_PLUGIN.value


def test_validate_returns_valid_flag_and_issues(client_and_service) -> None:
    # GIVEN a valid payload (stub returns [])
    client, _ = client_and_service
    givenPayload = {"stages": _valid_stages_payload()}

    # WHEN we call validate
    response = client.post("/v2/pipelines/validate", json=givenPayload)

    # THEN valid=true and no issues
    expectedStatus = 200
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["valid"] is True
    assert body["issues"] == []


def test_get_one_returns_the_pipeline(client_and_service) -> None:
    # GIVEN a persisted pipeline
    client, service = client_and_service
    givenDoc = PipelineDocument(
        pipeline_id="p1",
        user_id="local-user",
        name="A",
        stages=[StageDocument(plugin_id="tabiya.source.text.v1", config={})],
        is_active=False,
        is_default=False,
        is_readonly=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    service._docs = [givenDoc]

    # WHEN we GET by id
    response = client.get("/v2/pipelines/p1")

    # THEN 200 with the doc
    expectedStatus = 200
    assert response.status_code == expectedStatus
    assert response.json()["pipeline_id"] == "p1"


def test_get_one_returns_404_on_not_found(client_and_service) -> None:
    # GIVEN no rows
    client, _ = client_and_service

    # WHEN we get an unknown id
    response = client.get("/v2/pipelines/nope")

    # THEN 404
    expectedStatus = 404
    assert response.status_code == expectedStatus


def test_update_returns_updated_doc(client_and_service) -> None:
    # GIVEN a persisted pipeline
    client, service = client_and_service
    doc = PipelineDocument(
        pipeline_id="p1",
        user_id="local-user",
        name="A",
        stages=[StageDocument(plugin_id="tabiya.source.text.v1", config={})],
        is_active=False,
        is_default=False,
        is_readonly=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    service._docs = [doc]

    # WHEN we PUT
    response = client.put(
        "/v2/pipelines/p1",
        json={"name": "renamed", "stages": _valid_stages_payload()},
    )

    # THEN 200 with the new name
    expectedStatus = 200
    assert response.status_code == expectedStatus
    assert response.json()["name"] == "renamed"


def test_update_returns_409_on_readonly(client_and_service) -> None:
    # GIVEN a stub that raises ReadonlyPipelineError
    client, service = client_and_service
    service.raise_on["update"] = ReadonlyPipelineError("p1")

    # WHEN we PUT
    response = client.put(
        "/v2/pipelines/p1",
        json={"name": "renamed", "stages": _valid_stages_payload()},
    )

    # THEN 409
    expectedStatus = 409
    assert response.status_code == expectedStatus


def test_delete_returns_204(client_and_service) -> None:
    # GIVEN a persisted pipeline
    client, service = client_and_service
    doc = PipelineDocument(
        pipeline_id="p1",
        user_id="local-user",
        name="A",
        stages=[StageDocument(plugin_id="tabiya.source.text.v1", config={})],
        is_active=False,
        is_default=False,
        is_readonly=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    service._docs = [doc]

    # WHEN we DELETE
    response = client.delete("/v2/pipelines/p1")

    # THEN 204
    expectedStatus = 204
    assert response.status_code == expectedStatus


def test_activate_returns_updated_doc(client_and_service) -> None:
    # GIVEN a persisted pipeline
    client, service = client_and_service
    doc = PipelineDocument(
        pipeline_id="p1",
        user_id="local-user",
        name="A",
        stages=[StageDocument(plugin_id="tabiya.source.text.v1", config={})],
        is_active=False,
        is_default=False,
        is_readonly=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    service._docs = [doc]

    # WHEN we POST activate
    response = client.post("/v2/pipelines/p1/activate")

    # THEN 200 with is_active=True
    expectedStatus = 200
    assert response.status_code == expectedStatus
    assert response.json()["is_active"] is True


def test_activate_returns_422_on_validation_error(client_and_service) -> None:
    # GIVEN a stub that raises validation error
    client, service = client_and_service
    service.raise_on["activate"] = PipelineValidationError(
        [ValidationIssue(code=IssueCode.UNKNOWN_PLUGIN, message="x")]
    )

    # WHEN we POST activate
    response = client.post("/v2/pipelines/p1/activate")

    # THEN 422
    expectedStatus = 422
    assert response.status_code == expectedStatus


def test_clone_returns_201_with_the_new_id(client_and_service) -> None:
    # GIVEN a persisted pipeline
    client, service = client_and_service
    doc = PipelineDocument(
        pipeline_id="p1",
        user_id="local-user",
        name="A",
        stages=[StageDocument(plugin_id="tabiya.source.text.v1", config={})],
        is_active=False,
        is_default=False,
        is_readonly=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    service._docs = [doc]

    # WHEN we clone
    response = client.post("/v2/pipelines/p1/clone")

    # THEN 201 with a new pipeline_id and the copy suffix
    expectedStatus = 201
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["pipeline_id"] == "clone-id"
    assert body["name"].endswith("(copy)")

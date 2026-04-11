"""Tests for classify API routes."""

from http import HTTPStatus
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from classify.models import (
    Classification,
    ClassifiedEntity,
    ClassifyMetadata,
    ClassifyResponse,
    EntitySpan,
)
from classify.service import IClassifyService


def _make_classify_response() -> ClassifyResponse:
    return ClassifyResponse(
        classification=Classification(
            entities=[
                ClassifiedEntity(
                    entity_type="occupation",
                    surface_form="Head Chef",
                    span=EntitySpan(start=9, end=18),
                    linked_entities=[{"label": "chef", "similarity_score": 0.91}],
                )
            ],
            entity_counts={"occupation": 1},
        ),
        metadata=ClassifyMetadata(
            classifier_version="1.0.0",
            model_name="tabiya/roberta-base-job-ner",
            linker_model="all-MiniLM-L6-v2",
            processing_time_ms=60.0,
            input_text_hash="abc123",
        ),
    )


class TestClassifyRoute:
    @pytest.mark.asyncio
    async def test_classify_successful(self, client_with_mocks: tuple[TestClient, IClassifyService]):
        client, mock_service = client_with_mocks
        # GIVEN a valid request body with text
        given_text = "We need a Head Chef who can plan menus."

        # AND the service returns a valid classification result
        given_response = _make_classify_response()
        mock_service.classify = AsyncMock(return_value=given_response)

        # WHEN a POST request is made
        response = client.post("/v1/classify", json={"text": given_text})

        # THEN the response is OK
        assert response.status_code == HTTPStatus.OK

        # AND the response body matches the expected response
        assert response.json() == given_response.model_dump()

        # AND the service was called with the text and no options
        mock_service.classify.assert_called_once_with(given_text, None)

    @pytest.mark.asyncio
    async def test_classify_with_options(self, client_with_mocks: tuple[TestClient, IClassifyService]):
        client, mock_service = client_with_mocks
        # GIVEN a valid request with custom options
        given_text = "Senior software engineer needed."
        given_options = {"top_k": 3, "min_similarity": 0.6}

        # AND the service returns a valid response
        mock_service.classify = AsyncMock(return_value=_make_classify_response())

        # WHEN a POST request is made with options
        response = client.post("/v1/classify", json={"text": given_text, "options": given_options})

        # THEN the response is OK
        assert response.status_code == HTTPStatus.OK

        # AND the service was called with the options
        call_args = mock_service.classify.call_args
        assert call_args[0][1].top_k == 3
        assert call_args[0][1].min_similarity == 0.6

    @pytest.mark.asyncio
    async def test_classify_empty_body_returns_400(self, client_with_mocks: tuple[TestClient, IClassifyService]):
        client, mock_service = client_with_mocks
        # GIVEN a request body with no text, title, or description

        # WHEN a POST request is made
        response = client.post("/v1/classify", json={})

        # THEN the response is BAD REQUEST before the service is called
        assert response.status_code == HTTPStatus.BAD_REQUEST

    @pytest.mark.asyncio
    async def test_classify_text_too_long_returns_413(self, client_with_mocks: tuple[TestClient, IClassifyService]):
        client, mock_service = client_with_mocks
        # GIVEN a text that exceeds the maximum allowed length
        given_text = "x" * 100_001

        # WHEN a POST request is made
        response = client.post("/v1/classify", json={"text": given_text})

        # THEN the response is REQUEST ENTITY TOO LARGE before the service is called
        assert response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE

    @pytest.mark.asyncio
    async def test_classify_service_error_returns_502(self, client_with_mocks: tuple[TestClient, IClassifyService]):
        client, mock_service = client_with_mocks
        # GIVEN the service raises an exception (e.g. NER service unreachable)
        mock_service.classify = AsyncMock(side_effect=Exception("NER service unreachable"))

        # WHEN a POST request is made
        response = client.post("/v1/classify", json={"text": "some text"})

        # THEN the response is BAD GATEWAY
        assert response.status_code == HTTPStatus.BAD_GATEWAY


class TestBatchRoutes:
    _BATCH_JOBS = [{"job_id": "j1", "text": "Senior chef needed."}]

    @pytest.mark.asyncio
    async def test_submit_batch_returns_202(self, client_with_mocks: tuple[TestClient, IClassifyService]):
        client, mock_service = client_with_mocks
        mock_service.classify = AsyncMock(return_value=_make_classify_response())

        # WHEN a batch is submitted
        response = client.post("/v1/classify/batch", json={"jobs": self._BATCH_JOBS})

        # THEN accepted
        assert response.status_code == HTTPStatus.ACCEPTED
        body = response.json()
        assert body["status"] == "processing"
        assert body["total"] == 1
        assert "batch_id" in body

    @pytest.mark.asyncio
    async def test_submit_batch_too_large_returns_413(self, client_with_mocks: tuple[TestClient, IClassifyService]):
        client, _ = client_with_mocks
        # GIVEN more jobs than MAX_BATCH_SIZE (501 > 500)
        jobs = [{"job_id": f"j{i}", "text": "text"} for i in range(501)]

        response = client.post("/v1/classify/batch", json={"jobs": jobs})

        assert response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE

    @pytest.mark.asyncio
    async def test_batch_status_returns_processing(self, client_with_mocks: tuple[TestClient, IClassifyService]):
        client, mock_service = client_with_mocks
        mock_service.classify = AsyncMock(return_value=_make_classify_response())

        # GIVEN a submitted batch
        submit = client.post("/v1/classify/batch", json={"jobs": self._BATCH_JOBS})
        batch_id = submit.json()["batch_id"]

        # WHEN status is checked immediately
        response = client.get(f"/v1/batch/{batch_id}/status")

        # THEN it is found (processing or completed depending on task scheduling)
        assert response.status_code == HTTPStatus.OK
        assert response.json()["batch_id"] == batch_id
        assert response.json()["total"] == 1

    @pytest.mark.asyncio
    async def test_batch_status_not_found_returns_404(self, client_with_mocks: tuple[TestClient, IClassifyService]):
        client, _ = client_with_mocks

        response = client.get("/v1/batch/nonexistent-batch-id/status")

        assert response.status_code == HTTPStatus.NOT_FOUND

    @pytest.mark.asyncio
    async def test_batch_results_not_found_returns_404(self, client_with_mocks: tuple[TestClient, IClassifyService]):
        client, _ = client_with_mocks

        response = client.get("/v1/batch/nonexistent-batch-id/results")

        assert response.status_code == HTTPStatus.NOT_FOUND

    def test_batch_results_completed(self, client_with_mocks: tuple[TestClient, IClassifyService]):
        client, mock_service = client_with_mocks
        mock_service.classify = AsyncMock(return_value=_make_classify_response())

        # GIVEN a submitted batch
        submit = client.post("/v1/classify/batch", json={"jobs": self._BATCH_JOBS})
        batch_id = submit.json()["batch_id"]

        # Poll for completion (background task runs in the TestClient's event loop)
        import time
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            response = client.get(f"/v1/batch/{batch_id}/results")
            assert response.status_code == HTTPStatus.OK
            if response.json()["status"] == "completed":
                break
            time.sleep(0.1)

        # THEN results are present and batch is completed
        body = response.json()
        assert body["status"] == "completed"
        assert len(body["results"]) == 1
        assert body["results"][0]["job_id"] == "j1"
        assert body["results"][0]["status"] == "completed"

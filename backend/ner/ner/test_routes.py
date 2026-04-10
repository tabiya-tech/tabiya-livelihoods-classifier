"""Tests for NER API routes."""

from http import HTTPStatus
from typing import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from ner.models import Entity, EntitySpan, NERMetadata, NERResponse
from ner.service import INERService


class TestExtractEntitiesRoute:
    def test_extract_entities_successful(self, client_with_mocks: tuple[TestClient, INERService]):
        client, mock_service = client_with_mocks
        # GIVEN a valid request body
        given_text = "We need a Head Chef who can plan menus."

        # AND the service returns a valid response
        given_response = NERResponse(
            entities=[
                Entity(entity_type="occupation", surface_form="Head Chef", span=EntitySpan(start=9, end=18)),
                Entity(entity_type="skill", surface_form="plan menus", span=EntitySpan(start=33, end=43)),
            ],
            metadata=NERMetadata(model_name="tabiya/roberta-base-job-ner", entity_count=2, processing_time_ms=42.0),
        )
        mock_service.extract_entities = MagicMock(return_value=given_response)

        # WHEN a POST request is made
        response = client.post("/v1/ner", json={"text": given_text})

        # THEN the response is OK
        assert response.status_code == HTTPStatus.OK

        # AND the response body matches the expected response
        assert response.json() == given_response.model_dump()

        # AND the service was called with the correct arguments
        mock_service.extract_entities.assert_called_once_with(given_text, None)

    def test_extract_entities_with_entity_type_filter(self, client_with_mocks: tuple[TestClient, INERService]):
        client, mock_service = client_with_mocks
        # GIVEN a request filtered to occupation entities only
        given_text = "We need a Head Chef who can plan menus."
        given_entity_types = ["occupation"]

        # AND the service returns only occupation entities
        given_response = NERResponse(
            entities=[
                Entity(entity_type="occupation", surface_form="Head Chef", span=EntitySpan(start=9, end=18)),
            ],
            metadata=NERMetadata(model_name="tabiya/roberta-base-job-ner", entity_count=1, processing_time_ms=38.0),
        )
        mock_service.extract_entities = MagicMock(return_value=given_response)

        # WHEN a POST request is made with entity_types filter
        response = client.post("/v1/ner", json={"text": given_text, "entity_types": given_entity_types})

        # THEN the response is OK
        assert response.status_code == HTTPStatus.OK

        # AND the service was called with the entity_types filter
        mock_service.extract_entities.assert_called_once_with(given_text, given_entity_types)

    def test_extract_entities_empty_text_returns_400(self, client_with_mocks: tuple[TestClient, INERService]):
        client, mock_service = client_with_mocks
        # GIVEN the service raises ValueError for empty text
        mock_service.extract_entities = MagicMock(side_effect=ValueError("Field 'text' is required and cannot be empty"))

        # WHEN a POST request is made with empty text
        response = client.post("/v1/ner", json={"text": ""})

        # THEN the response is BAD REQUEST
        assert response.status_code == HTTPStatus.BAD_REQUEST

    def test_extract_entities_missing_text_returns_422(self, client_with_mocks: tuple[TestClient, INERService]):
        client, mock_service = client_with_mocks
        # GIVEN a request body with no text field

        # WHEN a POST request is made without the required text field
        response = client.post("/v1/ner", json={})

        # THEN Pydantic rejects it with UNPROCESSABLE ENTITY
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY

    def test_extract_entities_model_not_loaded_returns_503(self, client_with_mocks: tuple[TestClient, INERService]):
        client, mock_service = client_with_mocks
        # GIVEN the service raises RuntimeError because the model is not loaded
        mock_service.extract_entities = MagicMock(side_effect=RuntimeError("NER model is not loaded"))

        # WHEN a POST request is made
        response = client.post("/v1/ner", json={"text": "some text"})

        # THEN the response is SERVICE UNAVAILABLE
        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE

    def test_extract_entities_text_too_long_returns_413(self, client_with_mocks: tuple[TestClient, INERService]):
        client, mock_service = client_with_mocks
        # GIVEN a text that exceeds the maximum length
        given_text = "x" * 50001

        # WHEN a POST request is made with oversized text
        response = client.post("/v1/ner", json={"text": given_text})

        # THEN the response is REQUEST ENTITY TOO LARGE before the service is called
        assert response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE

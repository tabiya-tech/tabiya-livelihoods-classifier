"""Tests for NEL API routes."""

from http import HTTPStatus
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nel.models import LinkedEntity, NELMetadata, NELResponse, TaxonomyMatch
from nel.service import INELService


class TestLinkEntitiesRoute:
    def test_link_entities_successful(self, client_with_mocks: tuple[TestClient, INELService]):
        client, mock_service = client_with_mocks
        # GIVEN a valid request with one entity
        given_entities = [{"text": "Head Chef", "entity_type": "occupation"}]

        # AND the service returns a valid linked response
        given_response = NELResponse(
            linked_entities=[
                LinkedEntity(
                    input_text="Head Chef",
                    entity_type="occupation",
                    matches=[
                        TaxonomyMatch(similarity_score=0.91, taxonomy="esco", label="chef", code="3434.1")
                    ],
                )
            ],
            metadata=NELMetadata(linker_model="all-MiniLM-L6-v2", taxonomy="esco", processing_time_ms=18.0),
        )
        mock_service.link_entities = MagicMock(return_value=given_response)

        # WHEN a POST request is made
        response = client.post("/v1/nel", json={"entities": given_entities})

        # THEN the response is OK
        assert response.status_code == HTTPStatus.OK

        # AND the response body matches
        assert response.json() == given_response.model_dump()

        # AND the service was called with the serialized entities and no options
        mock_service.link_entities.assert_called_once_with(given_entities, None)

    def test_link_entities_with_options(self, client_with_mocks: tuple[TestClient, INELService]):
        client, mock_service = client_with_mocks
        # GIVEN a request with custom top_k and min_similarity options
        given_entities = [{"text": "Head Chef", "entity_type": "occupation"}]
        given_options = {"top_k": 3, "min_similarity": 0.5}

        # AND the service returns a valid response
        given_response = NELResponse(
            linked_entities=[],
            metadata=NELMetadata(linker_model="all-MiniLM-L6-v2", taxonomy="esco", processing_time_ms=5.0),
        )
        mock_service.link_entities = MagicMock(return_value=given_response)

        # WHEN a POST request is made with options
        response = client.post("/v1/nel", json={"entities": given_entities, "options": given_options})

        # THEN the response is OK
        assert response.status_code == HTTPStatus.OK

        # AND the service was called with the options
        call_args = mock_service.link_entities.call_args
        assert call_args[0][1].top_k == 3
        assert call_args[0][1].min_similarity == 0.5

    def test_link_entities_empty_list_returns_422(self, client_with_mocks: tuple[TestClient, INELService]):
        client, mock_service = client_with_mocks
        # GIVEN a request with an empty entities list

        # WHEN a POST request is made with empty entities
        response = client.post("/v1/nel", json={"entities": []})

        # THEN Pydantic rejects it with UNPROCESSABLE ENTITY (min_length=1)
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY

    def test_link_entities_missing_entities_returns_422(self, client_with_mocks: tuple[TestClient, INELService]):
        client, mock_service = client_with_mocks
        # GIVEN a request body with no entities field

        # WHEN a POST request is made without entities
        response = client.post("/v1/nel", json={})

        # THEN Pydantic rejects it with UNPROCESSABLE ENTITY
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY

    def test_link_entities_too_many_entities_returns_413(self, client_with_mocks: tuple[TestClient, INELService]):
        client, mock_service = client_with_mocks
        # GIVEN a request with more entities than allowed
        given_entities = [{"text": f"entity {i}", "entity_type": "skill"} for i in range(201)]

        # WHEN a POST request is made with too many entities
        response = client.post("/v1/nel", json={"entities": given_entities})

        # THEN the response is REQUEST ENTITY TOO LARGE before the service is called
        assert response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE

    def test_link_entities_linker_not_loaded_returns_503(self, client_with_mocks: tuple[TestClient, INELService]):
        client, mock_service = client_with_mocks
        # GIVEN the service raises RuntimeError because the linker is not loaded
        mock_service.link_entities = MagicMock(side_effect=RuntimeError("NEL linker is not loaded"))

        # WHEN a POST request is made
        response = client.post("/v1/nel", json={"entities": [{"text": "chef", "entity_type": "occupation"}]})

        # THEN the response is SERVICE UNAVAILABLE
        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE

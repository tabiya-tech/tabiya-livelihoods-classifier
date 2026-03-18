"""Tests that validate NEL API responses against nel-response.schema.json."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


SCHEMA_PATH = Path(__file__).parent.parent.parent / "api-specifications" / "nel-response.schema.json"


@pytest.fixture(scope="module")
def schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def client():
    mock_linker = MagicMock()
    mock_linker.similarity_model_name = "all-MiniLM-L6-v2"
    mock_linker.link.return_value = [
        {
            "input_text": "Head Chef",
            "entity_type": "occupation",
            "matches": [
                {
                    "similarity_score": 0.9123,
                    "taxonomy": "esco",
                    "label": "chef",
                    "code": "3434.1",
                    "uri": "http://data.europa.eu/esco/occupation/abc123",
                }
            ],
        }
    ]

    with patch("nel.main.nel_linker", mock_linker):
        from nel.main import app
        return TestClient(app)


def _validate_against_schema(data: dict, schema: dict) -> None:
    assert "linked_entities" in data, "Response missing 'linked_entities'"
    assert "metadata" in data, "Response missing 'metadata'"
    assert isinstance(data["linked_entities"], list)

    for item in data["linked_entities"]:
        assert "input_text" in item
        assert "entity_type" in item
        assert "matches" in item
        assert isinstance(item["matches"], list)

        for match in item["matches"]:
            assert "similarity_score" in match
            assert "taxonomy" in match
            assert "label" in match
            assert isinstance(match["similarity_score"], float)

    meta = data["metadata"]
    assert "linker_model" in meta
    assert "taxonomy" in meta
    assert "processing_time_ms" in meta


def test_nel_response_matches_schema(client, schema):
    response = client.post(
        "/v1/nel",
        json={"entities": [{"text": "Head Chef", "entity_type": "occupation"}]},
    )
    assert response.status_code == 200
    data = response.json()
    _validate_against_schema(data, schema)


def test_nel_empty_entities_returns_422(client):
    response = client.post("/v1/nel", json={"entities": []})
    assert response.status_code == 422


def test_nel_health_endpoint(client):
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "nel-api"

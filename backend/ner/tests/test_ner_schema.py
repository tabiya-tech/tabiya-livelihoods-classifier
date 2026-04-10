"""Tests that validate NER API responses against ner-response.schema.json."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


SCHEMA_PATH = Path(__file__).parent.parent.parent / "api-specifications" / "ner-response.schema.json"


@pytest.fixture(scope="module")
def schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def client():
    mock_model = MagicMock()
    mock_model.model_name = "tabiya/roberta-base-job-ner"
    mock_model.extract.return_value = [
        {
            "entity_type": "occupation",
            "surface_form": "Head Chef",
            "span": {"start": 9, "end": 18},
        },
        {
            "entity_type": "skill",
            "surface_form": "plan menus",
            "span": {"start": 33, "end": 43},
        },
    ]

    with patch("ner.model.NERModel", return_value=mock_model):
        import importlib
        import ner.main
        importlib.reload(ner.main)
        from ner.main import app
        return TestClient(app)


def _validate_against_schema(data: dict, schema: dict) -> None:
    """Minimal schema validation without jsonschema dependency."""
    assert "entities" in data, "Response missing 'entities'"
    assert "metadata" in data, "Response missing 'metadata'"
    assert isinstance(data["entities"], list), "'entities' must be a list"

    for entity in data["entities"]:
        assert "entity_type" in entity, "Entity missing 'entity_type'"
        assert "surface_form" in entity, "Entity missing 'surface_form'"
        assert "span" in entity, "Entity missing 'span'"
        assert "start" in entity["span"], "Span missing 'start'"
        assert "end" in entity["span"], "Span missing 'end'"
        assert entity["entity_type"] in (
            "occupation", "skill", "qualification", "experience", "domain"
        ), f"Unknown entity_type: {entity['entity_type']}"

    meta = data["metadata"]
    assert "model_name" in meta, "Metadata missing 'model_name'"
    assert "entity_count" in meta, "Metadata missing 'entity_count'"
    assert "processing_time_ms" in meta, "Metadata missing 'processing_time_ms'"
    assert meta["entity_count"] == len(data["entities"]), "entity_count mismatch"


def test_ner_response_matches_schema(client, schema):
    response = client.post("/v1/ner", json={"text": "We need a Head Chef who can plan menus."})
    assert response.status_code == 200
    data = response.json()
    _validate_against_schema(data, schema)


def test_ner_response_entity_types_filter(client, schema):
    response = client.post(
        "/v1/ner",
        json={"text": "We need a Head Chef who can plan menus.", "entity_types": ["occupation"]},
    )
    assert response.status_code == 200
    data = response.json()
    _validate_against_schema(data, schema)
    for entity in data["entities"]:
        assert entity["entity_type"] == "occupation"


def test_ner_empty_text_returns_400(client):
    response = client.post("/v1/ner", json={"text": ""})
    assert response.status_code == 400


def test_ner_health_endpoint(client):
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ner-api"

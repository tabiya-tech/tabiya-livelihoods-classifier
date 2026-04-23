"""Tests for classify v2 service."""

import pytest
import respx
import httpx

from classify_v2.app.classification.service.errors import EmbeddingsCacheNotReadyError, NERServiceError, NELServiceError
from classify_v2.app.classification.service.service import ClassifyService
from classify_v2.app.classification.service.types import ClassifyOptions


_NER_URL = "http://ner-test"
_NEL_URL = "http://nel-test"

_NER_RESPONSE = {
    "entities": [
        {"surface_form": "Head Chef", "entity_type": "occupation", "span": {"start": 0, "end": 9}},
        {"surface_form": "Python", "entity_type": "skill", "span": {"start": 10, "end": 16}},
    ],
    "metadata": {"model_name": "ner-model-v1"},
}

_NEL_RESPONSE = {
    "linked_entities": [
        {
            "input_text": "Head Chef",
            "entity_type": "occupation",
            "matches": [
                {
                    "entity_type": "occupation",
                    "similarity_score": 0.92,
                    "entity": {
                        "uuid": "u1",
                        "origin_uuid": "u1",
                        "uuid_history": ["u1"],
                        "preferred_label": "Head Chef",
                        "origin_uri": "http://example.com",
                        "alt_labels": [],
                        "description": "",
                        "esco_code": "1234.1",
                    },
                }
            ],
        },
        {
            "input_text": "Python",
            "entity_type": "skill",
            "matches": [
                {
                    "entity_type": "skill",
                    "similarity_score": 0.85,
                    "entity": {
                        "uuid": "u2",
                        "origin_uuid": "u2",
                        "uuid_history": ["u2"],
                        "preferred_label": "Python (programming language)",
                        "origin_uri": "http://example.com/skill",
                        "alt_labels": [],
                        "description": "",
                        "skill_type": "skill/competence",
                        "reuse_level": "cross-sector",
                    },
                }
            ],
        },
    ],
    "metadata": {"nel_model_id": "all-MiniLM-L6-v2", "taxonomy_model_id": "tax-1", "processing_time_ms": 50.0},
}


class TestClassifyService:
    @respx.mock
    async def test_happy_path_returns_classified_entities(self):
        # GIVEN NER and NEL both respond successfully
        respx.post(f"{_NER_URL}/v1/ner").mock(return_value=httpx.Response(200, json=_NER_RESPONSE))
        respx.post(f"{_NEL_URL}/v2/nel").mock(return_value=httpx.Response(200, json=_NEL_RESPONSE))
        svc = ClassifyService(ner_api_url=_NER_URL, nel_v2_api_url=_NEL_URL)

        # WHEN classify is called
        result = await svc.classify("Head Chef Python", firebase_token="test-token")

        # THEN both entities are returned with their matches
        assert len(result.entities) == 2
        occupation = next(e for e in result.entities if e.entity_type == "occupation")
        assert occupation.surface_form == "Head Chef"
        assert len(occupation.matches) == 1
        assert occupation.matches[0].entity.preferred_label == "Head Chef"

    @respx.mock
    async def test_metadata_propagated_from_nel_response(self):
        # GIVEN NER and NEL respond with known metadata
        respx.post(f"{_NER_URL}/v1/ner").mock(return_value=httpx.Response(200, json=_NER_RESPONSE))
        respx.post(f"{_NEL_URL}/v2/nel").mock(return_value=httpx.Response(200, json=_NEL_RESPONSE))
        svc = ClassifyService(ner_api_url=_NER_URL, nel_v2_api_url=_NEL_URL)

        # WHEN classify is called
        result = await svc.classify("Head Chef Python", firebase_token="test-token")

        # THEN metadata reflects the nel model and taxonomy used
        assert result.metadata.nel_model_id == "all-MiniLM-L6-v2"
        assert result.metadata.taxonomy_model_id == "tax-1"
        assert result.metadata.ner_model == "ner-model-v1"

    @respx.mock
    async def test_nel_not_called_when_no_linkable_entities(self):
        # GIVEN NER returns no linkable entity types
        ner_response = {
            "entities": [],
            "metadata": {"model_name": "ner-model-v1"},
        }
        respx.post(f"{_NER_URL}/v1/ner").mock(return_value=httpx.Response(200, json=ner_response))
        nel_mock = respx.post(f"{_NEL_URL}/v2/nel").mock(return_value=httpx.Response(200, json={"linked_entities": [], "metadata": {}}))
        svc = ClassifyService(ner_api_url=_NER_URL, nel_v2_api_url=_NEL_URL)

        # WHEN classify is called
        result = await svc.classify("some text", firebase_token="test-token")

        # THEN NEL is not called and result has no entities
        assert not nel_mock.called
        assert result.entities == []

    @respx.mock
    async def test_raises_ner_service_error_on_ner_failure(self):
        # GIVEN NER returns a 500
        respx.post(f"{_NER_URL}/v1/ner").mock(return_value=httpx.Response(500, text="internal error"))
        svc = ClassifyService(ner_api_url=_NER_URL, nel_v2_api_url=_NEL_URL)

        # WHEN classify is called
        # THEN NERServiceError is raised
        with pytest.raises(NERServiceError):
            await svc.classify("Head Chef", firebase_token="test-token")

    @respx.mock
    async def test_raises_embeddings_cache_not_ready_on_503(self):
        # GIVEN NER succeeds but NEL returns 503
        respx.post(f"{_NER_URL}/v1/ner").mock(return_value=httpx.Response(200, json=_NER_RESPONSE))
        respx.post(f"{_NEL_URL}/v2/nel").mock(return_value=httpx.Response(503, json={"detail": "Embeddings not ready"}))
        svc = ClassifyService(ner_api_url=_NER_URL, nel_v2_api_url=_NEL_URL)

        # WHEN classify is called
        # THEN EmbeddingsCacheNotReadyError is raised
        with pytest.raises(EmbeddingsCacheNotReadyError):
            await svc.classify("Head Chef", firebase_token="test-token")

    @respx.mock
    async def test_firebase_token_forwarded_to_nel(self):
        # GIVEN NER and NEL both respond successfully
        respx.post(f"{_NER_URL}/v1/ner").mock(return_value=httpx.Response(200, json=_NER_RESPONSE))
        nel_mock = respx.post(f"{_NEL_URL}/v2/nel").mock(return_value=httpx.Response(200, json=_NEL_RESPONSE))
        svc = ClassifyService(ner_api_url=_NER_URL, nel_v2_api_url=_NEL_URL)

        # WHEN classify is called with a firebase token
        await svc.classify("Head Chef", firebase_token="my-firebase-token")

        # THEN the token is forwarded to the NEL service
        assert nel_mock.calls[0].request.headers["authorization"] == "Bearer my-firebase-token"

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nel.app.embedding.service.service import (
    SentenceTransformerEmbeddingService,
    _clear_registry,
    get_embedding_service,
)


def _make_mock_model(dimensions: int = 384):
    """Return a SentenceTransformer mock that returns fixed-size numpy arrays."""
    mock = MagicMock()
    mock.get_sentence_embedding_dimension.return_value = dimensions
    mock.encode.side_effect = lambda texts, **_kwargs: np.zeros((len(texts), dimensions))
    return mock


@pytest.fixture(autouse=True)
def clear_registry():
    _clear_registry()
    yield
    _clear_registry()


# ── SentenceTransformerEmbeddingService ───────────────────────────────────

class TestSentenceTransformerEmbeddingService:
    def test_dimensions_inferred_from_model(self):
        # GIVEN a mock model with 384 dimensions
        mock_model = _make_mock_model(384)
        with patch("nel.app.embedding.service.service.SentenceTransformer", return_value=mock_model):
            # WHEN the service is instantiated
            svc = SentenceTransformerEmbeddingService("all-MiniLM-L6-v2")

        # THEN dimensions are taken from the model
        assert svc.dimensions == 384
        assert svc.model_id == "all-MiniLM-L6-v2"

    async def test_embed_batch_calls_encode_with_all_texts(self):
        # GIVEN a mock model
        mock_model = _make_mock_model(384)
        with patch("nel.app.embedding.service.service.SentenceTransformer", return_value=mock_model):
            svc = SentenceTransformerEmbeddingService("all-MiniLM-L6-v2")

        # WHEN embed_batch is called with two texts
        texts = ["Head Chef", "Python Developer"]
        results = await svc.embed_batch(texts)

        # THEN encode was called once with all texts
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args
        assert call_args.args[0] == texts

        # AND the output shape matches
        assert len(results) == 2
        assert len(results[0]) == 384

    async def test_embed_returns_single_vector(self):
        # GIVEN a mock model
        mock_model = _make_mock_model(384)
        with patch("nel.app.embedding.service.service.SentenceTransformer", return_value=mock_model):
            svc = SentenceTransformerEmbeddingService("all-MiniLM-L6-v2")

        # WHEN embed is called with a single text
        result = await svc.embed("Head Chef")

        # THEN a single flat list of floats is returned
        assert isinstance(result, list)
        assert len(result) == 384


# ── Singleton registry ────────────────────────────────────────────────────

class TestGetEmbeddingService:
    async def test_same_instance_returned_for_same_model_id(self):
        # GIVEN the registry is empty
        mock_model = _make_mock_model()
        with patch("nel.app.embedding.service.service.SentenceTransformer", return_value=mock_model):
            # WHEN get_embedding_service is called twice with the same model_id
            svc1 = await get_embedding_service("all-MiniLM-L6-v2")
            svc2 = await get_embedding_service("all-MiniLM-L6-v2")

        # THEN the same instance is returned (model loaded only once)
        assert svc1 is svc2

    async def test_different_instances_for_different_model_ids(self):
        # GIVEN the registry is empty
        mock_model = _make_mock_model()
        with patch("nel.app.embedding.service.service.SentenceTransformer", return_value=mock_model):
            # WHEN get_embedding_service is called with two different model IDs
            svc1 = await get_embedding_service("model-a")
            svc2 = await get_embedding_service("model-b")

        # THEN different instances are returned
        assert svc1 is not svc2

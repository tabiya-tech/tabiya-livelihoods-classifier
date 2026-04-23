from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from nel.app.embedding.service.service import (
    GoogleVertexEmbeddingService,
    SentenceTransformerEmbeddingService,
    _clear_registry,
    get_embedding_service,
)


def _make_st_mock(dimensions: int = 384):
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
        mock_model = _make_st_mock(384)
        with patch("nel.app.embedding.service.service.SentenceTransformer", return_value=mock_model):
            # WHEN the service is instantiated
            svc = SentenceTransformerEmbeddingService("all-MiniLM-L6-v2")

        # THEN dimensions are taken from the model
        assert svc.dimensions == 384
        assert svc.model_id == "all-MiniLM-L6-v2"

    async def test_embed_batch_calls_encode_with_all_texts(self):
        # GIVEN a mock model
        mock_model = _make_st_mock(384)
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
        mock_model = _make_st_mock(384)
        with patch("nel.app.embedding.service.service.SentenceTransformer", return_value=mock_model):
            svc = SentenceTransformerEmbeddingService("all-MiniLM-L6-v2")

        # WHEN embed is called with a single text
        result = await svc.embed("Head Chef")

        # THEN a single flat list of floats is returned
        assert isinstance(result, list)
        assert len(result) == 384


# ── GoogleVertexEmbeddingService ──────────────────────────────────────────

def _make_vertex_mocks():
    """Return (mock_vertexai, mock_TextEmbeddingModel, mock_TextEmbeddingInput)."""
    mock_vertexai = MagicMock()
    mock_model_instance = MagicMock()
    mock_TextEmbeddingModel = MagicMock()
    mock_TextEmbeddingModel.from_pretrained.return_value = mock_model_instance
    mock_TextEmbeddingInput = MagicMock(side_effect=lambda text, task: MagicMock())
    return mock_vertexai, mock_TextEmbeddingModel, mock_TextEmbeddingInput, mock_model_instance


class TestGoogleVertexEmbeddingService:
    def _make_svc(self, mock_model_instance, dims: int = 768):
        mock_vertexai = MagicMock()
        mock_TextEmbeddingModel = MagicMock()
        mock_TextEmbeddingModel.from_pretrained.return_value = mock_model_instance
        mock_TextEmbeddingInput = MagicMock(side_effect=lambda text, task: MagicMock())
        with (
            patch("nel.app.embedding.service.service.vertexai", mock_vertexai, create=True),
            patch.dict("sys.modules", {
                "vertexai": mock_vertexai,
                "vertexai.language_models": MagicMock(
                    TextEmbeddingModel=mock_TextEmbeddingModel,
                    TextEmbeddingInput=mock_TextEmbeddingInput,
                ),
            }),
        ):
            svc = GoogleVertexEmbeddingService("text-embedding-005", "us-central1")
        svc._model = mock_model_instance
        svc._region = "us-central1"
        return svc

    async def test_embed_batch_returns_correct_shape(self):
        # GIVEN a Vertex model mock that returns 768-d embeddings
        dims = 768
        mock_model_instance = MagicMock()
        mock_result = [MagicMock(values=[0.0] * dims) for _ in range(2)]
        mock_model_instance.get_embeddings_async = AsyncMock(return_value=mock_result)

        svc = self._make_svc(mock_model_instance, dims)

        # WHEN embed_batch is called with two texts
        with patch.dict("sys.modules", {
            "vertexai": MagicMock(),
            "vertexai.language_models": MagicMock(
                TextEmbeddingInput=MagicMock(side_effect=lambda text, task: MagicMock()),
            ),
        }):
            results = await svc.embed_batch(["Head Chef", "Python Developer"])

        # THEN the output has two 768-d vectors
        assert len(results) == 2
        assert len(results[0]) == dims

    async def test_embed_returns_single_vector(self):
        # GIVEN a Vertex model mock
        dims = 768
        mock_model_instance = MagicMock()
        mock_model_instance.get_embeddings_async = AsyncMock(
            return_value=[MagicMock(values=[0.1] * dims)]
        )
        svc = self._make_svc(mock_model_instance, dims)

        # WHEN embed is called with a single text
        with patch.dict("sys.modules", {
            "vertexai": MagicMock(),
            "vertexai.language_models": MagicMock(
                TextEmbeddingInput=MagicMock(side_effect=lambda text, task: MagicMock()),
            ),
        }):
            result = await svc.embed("Head Chef")

        # THEN a single 768-d vector is returned
        assert isinstance(result, list)
        assert len(result) == dims

    def test_dimensions_and_model_id_set_correctly(self):
        # GIVEN a GoogleVertexEmbeddingService
        mock_model_instance = MagicMock()
        svc = self._make_svc(mock_model_instance)

        # THEN model_id and dimensions are set from the known mapping
        assert svc.model_id == "text-embedding-005"
        assert svc.dimensions == 768


# ── Singleton registry ────────────────────────────────────────────────────

class TestGetEmbeddingService:
    async def test_same_instance_returned_for_same_model_id(self):
        # GIVEN the registry is empty
        mock_model = _make_st_mock()
        with patch("nel.app.embedding.service.service.SentenceTransformer", return_value=mock_model):
            # WHEN get_embedding_service is called twice with the same model_id
            svc1 = await get_embedding_service("all-MiniLM-L6-v2")
            svc2 = await get_embedding_service("all-MiniLM-L6-v2")

        # THEN the same instance is returned (model loaded only once)
        assert svc1 is svc2

    async def test_different_instances_for_different_model_ids(self):
        # GIVEN the registry is empty
        mock_model = _make_st_mock()
        with patch("nel.app.embedding.service.service.SentenceTransformer", return_value=mock_model):
            # WHEN get_embedding_service is called with two different model IDs
            svc1 = await get_embedding_service("model-a")
            svc2 = await get_embedding_service("model-b")

        # THEN different instances are returned
        assert svc1 is not svc2

    async def test_vertex_model_id_routes_to_vertex_service(self):
        # GIVEN VERTEX_API_REGION is set
        mock_vertex = MagicMock()
        mock_model_instance = MagicMock()
        mock_vertex.language_models.TextEmbeddingModel.from_pretrained.return_value = mock_model_instance
        with (
            patch.dict("sys.modules", {
                "vertexai": mock_vertex,
                "vertexai.language_models": MagicMock(
                    TextEmbeddingModel=MagicMock(from_pretrained=MagicMock(return_value=mock_model_instance)),
                    TextEmbeddingInput=MagicMock(),
                ),
            }),
            patch.dict("os.environ", {"VERTEX_API_REGION": "us-central1"}),
        ):
            # WHEN get_embedding_service is called for a Vertex model
            svc = await get_embedding_service("text-embedding-005")

        # THEN a GoogleVertexEmbeddingService is returned
        assert isinstance(svc, GoogleVertexEmbeddingService)

    async def test_gemini_model_id_routes_to_vertex_service(self):
        # GIVEN VERTEX_API_REGION is set (gemini-embedding-001 uses Vertex AI, not google-generativeai)
        mock_model_instance = MagicMock()
        with (
            patch.dict("sys.modules", {
                "vertexai": MagicMock(),
                "vertexai.language_models": MagicMock(
                    TextEmbeddingModel=MagicMock(from_pretrained=MagicMock(return_value=mock_model_instance)),
                    TextEmbeddingInput=MagicMock(),
                ),
            }),
            patch.dict("os.environ", {"VERTEX_API_REGION": "us-central1"}),
        ):
            # WHEN get_embedding_service is called for the Gemini embedding model
            svc = await get_embedding_service("models/gemini-embedding-001")

        # THEN a GoogleVertexEmbeddingService is returned (ADC auth, no API key needed)
        assert isinstance(svc, GoogleVertexEmbeddingService)
        assert svc.dimensions == 3072

    async def test_missing_vertex_region_raises(self):
        # GIVEN VERTEX_API_REGION is not set
        with (
            patch.dict("sys.modules", {
                "vertexai": MagicMock(),
                "vertexai.language_models": MagicMock(),
            }),
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="VERTEX_API_REGION"),
        ):
            # WHEN get_embedding_service is called for a Vertex model
            await get_embedding_service("text-embedding-005")

"""Embedding service — converts text to float vectors.

Two implementations:
  SentenceTransformerEmbeddingService — local HuggingFace models via sentence-transformers
  GoogleVertexEmbeddingService        — Google Vertex AI (text-embedding-005 and
                                        models/gemini-embedding-001), authenticated via
                                        Application Default Credentials (ADC).

get_embedding_service() is a singleton registry keyed by model_id; each model is
loaded or initialised once and reused for the lifetime of the process.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from sentence_transformers import SentenceTransformer

_logger = logging.getLogger(__name__)

# Model IDs handled by Google Vertex AI (TextEmbeddingModel).
# Both use ADC — no API key required.
_VERTEX_MODEL_IDS: frozenset[str] = frozenset({
    "text-embedding-005",
    "models/gemini-embedding-001",
})


# ── Interface ─────────────────────────────────────────────────────────────

class IEmbeddingService(ABC):
    model_id: str
    dimensions: int

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings in one call."""


# ── SentenceTransformer implementation ────────────────────────────────────

class SentenceTransformerEmbeddingService(IEmbeddingService):
    def __init__(self, model_id: str):
        self.model_id = model_id
        _logger.info("Loading SentenceTransformer model: %s", model_id)
        self._model = SentenceTransformer(model_id)
        self.dimensions: int = self._model.get_sentence_embedding_dimension()
        # One executor per service instance; keeps CPU work off the event loop.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"embed-{model_id}")
        _logger.info("Loaded model %s (dimensions=%d)", model_id, self.dimensions)

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        encode_fn = partial(self._model.encode, texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = await loop.run_in_executor(self._executor, encode_fn)
        return [emb.tolist() for emb in embeddings]


# ── Google Vertex AI implementation ───────────────────────────────────────

# Per-model batch size limits.
# text-embedding-005: up to 250 in us-central1, 5 elsewhere.
# gemini-embedding-001: max 5 regardless of region (3072-d vectors are large).
# https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings#supported-models
_VERTEX_BATCH_SIZE_DEFAULT = 5

_VERTEX_DIMENSIONS = {
    "text-embedding-005": 768,
    "models/gemini-embedding-001": 3072,
}

# Models that support a larger batch size when in us-central1.
_VERTEX_LARGE_BATCH_MODELS: frozenset[str] = frozenset({"text-embedding-005"})
_VERTEX_BATCH_SIZE_US_CENTRAL1 = 250


class GoogleVertexEmbeddingService(IEmbeddingService):
    """Wraps Google Vertex AI TextEmbeddingModel (text-embedding-005).

    Requires the VERTEX_API_REGION environment variable.
    Uses RETRIEVAL_DOCUMENT task type for generating stored embeddings and
    RETRIEVAL_QUERY for query embeddings. For simplicity (and consistency with
    Compass) we use RETRIEVAL_QUERY for both here since at query time it matches
    how Compass generates its stored embeddings.
    """

    _TASK = "RETRIEVAL_QUERY"

    def __init__(self, model_id: str, region: str):
        import vertexai
        from vertexai.language_models import TextEmbeddingModel

        self.model_id = model_id
        self.dimensions = _VERTEX_DIMENSIONS[model_id]
        self._region = region
        # Vertex AI SDK does not accept the "models/" prefix that the Generative AI SDK uses.
        vertex_model_name = model_id.removeprefix("models/")
        vertexai.init(location=region)
        self._model = TextEmbeddingModel.from_pretrained(vertex_model_name)
        _logger.info("Initialised GoogleVertexEmbeddingService: model=%s region=%s dims=%d", model_id, region, self.dimensions)

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import vertexai
        from vertexai.language_models import TextEmbeddingInput

        vertexai.init(location=self._region)
        if self.model_id in _VERTEX_LARGE_BATCH_MODELS and self._region == "us-central1":
            batch_size = _VERTEX_BATCH_SIZE_US_CENTRAL1
        else:
            batch_size = _VERTEX_BATCH_SIZE_DEFAULT
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start: start + batch_size]
            inputs = [TextEmbeddingInput(text, self._TASK) for text in chunk]
            results = await asyncio.wait_for(self._model.get_embeddings_async(inputs), timeout=120)
            all_embeddings.extend(result.values for result in results)
        return all_embeddings


# ── Singleton registry ────────────────────────────────────────────────────

_service_registry: dict[str, IEmbeddingService] = {}
_registry_lock = asyncio.Lock()


async def get_embedding_service(model_id: str) -> IEmbeddingService:
    """Return a cached IEmbeddingService for model_id, creating it if needed.

    Routing:
      - model_id in _VERTEX_MODEL_IDS  → GoogleVertexEmbeddingService (needs VERTEX_API_REGION)
      - model_id in _GEMINI_MODEL_IDS  → GoogleGeminiEmbeddingService (needs GOOGLE_API_KEY)
      - anything else                  → SentenceTransformerEmbeddingService
    """
    if model_id not in _service_registry:
        async with _registry_lock:
            if model_id not in _service_registry:
                _service_registry[model_id] = _create_embedding_service(model_id)
    return _service_registry[model_id]


def _create_embedding_service(model_id: str) -> IEmbeddingService:
    if model_id in _VERTEX_MODEL_IDS:
        import os
        region = os.getenv("VERTEX_API_REGION")
        if not region:
            raise ValueError(f"VERTEX_API_REGION env var is required for Vertex AI model '{model_id}'")
        return GoogleVertexEmbeddingService(model_id, region)
    return SentenceTransformerEmbeddingService(model_id)


def _clear_registry() -> None:
    """Reset the registry. Used in tests."""
    _service_registry.clear()

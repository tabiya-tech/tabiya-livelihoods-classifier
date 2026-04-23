"""Embedding service — converts text to float vectors.

SentenceTransformerEmbeddingService wraps SentenceTransformer and dispatches
encode() calls to a ThreadPoolExecutor so the async event loop is never blocked.

Singleton registry (get_embedding_service) ensures each model is loaded once.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from sentence_transformers import SentenceTransformer

_logger = logging.getLogger(__name__)


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


# ── Singleton registry ────────────────────────────────────────────────────

_service_registry: dict[str, IEmbeddingService] = {}
_registry_lock = asyncio.Lock()


async def get_embedding_service(model_id: str) -> IEmbeddingService:
    """Return a cached IEmbeddingService for model_id, creating it if needed."""
    if model_id not in _service_registry:
        async with _registry_lock:
            if model_id not in _service_registry:
                _service_registry[model_id] = SentenceTransformerEmbeddingService(model_id)
    return _service_registry[model_id]


def _clear_registry() -> None:
    """Reset the registry. Used in tests."""
    _service_registry.clear()

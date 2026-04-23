class EmbeddingError(Exception):
    """Base error for embedding service failures."""


class ModelNotFoundError(EmbeddingError):
    """Raised when a requested embedding model ID is not available."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Embedding model not found: {model_id!r}")

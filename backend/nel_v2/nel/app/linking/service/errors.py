class EmbeddingsCacheNotReadyError(Exception):
    """Raised when vector search is attempted but embeddings have not been generated yet."""

    def __init__(self, taxonomy_model_id: str, nel_model_id: str, status: str):
        self.taxonomy_model_id = taxonomy_model_id
        self.nel_model_id = nel_model_id
        self.status = status
        super().__init__(
            f"Embeddings cache not ready for taxonomy_model_id={taxonomy_model_id!r}, "
            f"nel_model_id={nel_model_id!r} (status={status!r}). "
            f"Trigger generation via POST /v2/nel/embeddings/generate."
        )

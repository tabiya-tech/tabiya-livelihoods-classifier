class NERServiceError(Exception):
    """Raised when the NER service returns an error or is unreachable."""


class NELServiceError(Exception):
    """Raised when the NEL v2 service returns an error or is unreachable."""


class EmbeddingsCacheNotReadyError(NELServiceError):
    """Raised when nel_v2 returns 503 — embeddings not yet generated."""

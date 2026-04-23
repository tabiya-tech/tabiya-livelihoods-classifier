from enum import Enum
from typing import Optional

from pydantic import BaseModel


class CacheStatus(str, Enum):
    pending = "pending"
    generating = "generating"
    ready = "ready"
    failed = "failed"


class EmbeddingCacheStatus(BaseModel):
    taxonomy_model_id: str
    nel_model_id: str
    entity_type: str  # "occupation" | "skill" | "qualification"
    status: CacheStatus
    total_count: int = 0
    error_message: Optional[str] = None


class EmbeddingDocument(BaseModel):
    """A single embedded taxonomy entity stored in MongoDB."""
    # Identity
    taxonomy_model_id: str
    nel_model_id: str
    entity_type: str
    # Taxonomy fields (mirrors TaxonomyMatch minus similarity_score)
    entity_uuid: str
    origin_uuid: str
    uuid_history: list[str]
    preferred_label: str
    origin_uri: str
    alt_labels: list[str]
    description: str
    # Type-specific
    esco_code: Optional[str] = None
    skill_type: Optional[str] = None
    reuse_level: Optional[str] = None
    eqf_level: Optional[str] = None
    country: Optional[str] = None
    # Embedding
    embedded_text: str
    embedding: list[float]

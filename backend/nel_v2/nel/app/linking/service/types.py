from enum import Enum
from typing import Optional

from pydantic import BaseModel


class EntityType(str, Enum):
    occupation = "occupation"
    skill = "skill"
    qualification = "qualification"


class TaxonomyMatch(BaseModel):
    similarity_score: float
    taxonomy_model_id: str
    nel_model_id: str
    # Core taxonomy identity
    entity_uuid: str
    origin_uuid: str
    uuid_history: list[str]
    preferred_label: str
    origin_uri: str
    alt_labels: list[str]
    description: str
    # Occupation-specific
    esco_code: Optional[str] = None
    # Skill-specific
    skill_type: Optional[str] = None
    reuse_level: Optional[str] = None
    # Qualification-specific
    eqf_level: Optional[str] = None
    country: Optional[str] = None


class LinkedEntity(BaseModel):
    input_text: str
    entity_type: EntityType
    matches: list[TaxonomyMatch]


class NELMetadata(BaseModel):
    nel_model_id: str
    taxonomy_model_id: str
    processing_time_ms: float


class NELResponse(BaseModel):
    linked_entities: list[LinkedEntity]
    metadata: NELMetadata


class NELOptions(BaseModel):
    top_k: int = 5
    min_similarity: float = 0.0

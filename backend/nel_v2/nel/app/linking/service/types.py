from enum import Enum
from typing import Annotated, Optional, Union

from pydantic import BaseModel


class EntityType(str, Enum):
    occupation = "occupation"
    skill = "skill"
    qualification = "qualification"


# ── Base entity fields shared by all types ────────────────────────────────────

class _BaseEntity(BaseModel):
    uuid: str
    origin_uuid: str
    uuid_history: list[str]
    preferred_label: str
    origin_uri: str
    alt_labels: list[str]
    description: str


# ── Typed entity models ───────────────────────────────────────────────────────

class OccupationEntity(_BaseEntity):
    esco_code: Optional[str] = None


class SkillEntity(_BaseEntity):
    skill_type: Optional[str] = None
    reuse_level: Optional[str] = None


class QualificationEntity(_BaseEntity):
    eqf_level: Optional[str] = None
    country: Optional[str] = None


# ── Typed match models (discriminated union on entity_type) ───────────────────

class OccupationMatch(BaseModel):
    entity_type: EntityType = EntityType.occupation
    similarity_score: float
    entity: OccupationEntity


class SkillMatch(BaseModel):
    entity_type: EntityType = EntityType.skill
    similarity_score: float
    entity: SkillEntity


class QualificationMatch(BaseModel):
    entity_type: EntityType = EntityType.qualification
    similarity_score: float
    entity: QualificationEntity


TaxonomyMatch = Annotated[
    Union[OccupationMatch, SkillMatch, QualificationMatch],
    "discriminated by entity_type"
]


# ── Response types ────────────────────────────────────────────────────────────

class LinkedEntity(BaseModel):
    input_text: str
    entity_type: EntityType
    matches: list[OccupationMatch | SkillMatch | QualificationMatch]


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

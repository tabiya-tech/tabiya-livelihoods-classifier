"""Classify v2 request/response types."""

from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    occupation = "occupation"
    skill = "skill"
    qualification = "qualification"


class ClassifyOptions(BaseModel):
    extract_entities: Optional[list[EntityType]] = Field(
        None,
        description="Restrict extraction to specific entity types. Omit to extract all.",
    )
    top_k: int = Field(5, ge=1, le=50)
    min_similarity: float = Field(0.0, ge=0.0, le=1.0)


class ClassifyRequest(BaseModel):
    text: Optional[str] = Field(None, description="Raw job ad text.")
    title: Optional[str] = Field(None, description="Job title.")
    description: Optional[str] = Field(None, description="Job description.")
    options: Optional[ClassifyOptions] = None


# ── Entity types (mirrors nel_v2 types) ──────────────────────────────────────

class _BaseEntity(BaseModel):
    uuid: str
    origin_uuid: str
    uuid_history: list[str]
    preferred_label: str
    origin_uri: str
    alt_labels: list[str]
    description: str


class OccupationEntity(_BaseEntity):
    esco_code: Optional[str] = None


class SkillEntity(_BaseEntity):
    skill_type: Optional[str] = None
    reuse_level: Optional[str] = None


class QualificationEntity(_BaseEntity):
    eqf_level: Optional[str] = None
    country: Optional[str] = None


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


TaxonomyMatch = Union[OccupationMatch, SkillMatch, QualificationMatch]


class EntitySpan(BaseModel):
    start: int
    end: int


class ClassifiedEntity(BaseModel):
    entity_type: str
    surface_form: str
    span: EntitySpan
    matches: list[TaxonomyMatch] = []


class ClassifyMetadata(BaseModel):
    classifier_version: str
    ner_model: str
    nel_model_id: str
    taxonomy_model_id: str
    processing_time_ms: float


class ClassifyResponse(BaseModel):
    entities: list[ClassifiedEntity]
    metadata: ClassifyMetadata

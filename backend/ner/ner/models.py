"""Pydantic request/response models for the NER API."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    occupation = "occupation"
    skill = "skill"
    qualification = "qualification"


class NERRequest(BaseModel):
    text: str = Field(..., description="Job-related text to extract entities from.")
    entity_types: Optional[List[EntityType]] = Field(
        None,
        description=(
            "Filter results to specific entity types. "
            "Omit to return all types. "
            "Allowed values: 'occupation', 'skill', 'qualification'."
        ),
    )


class EntitySpan(BaseModel):
    start: int
    end: int


class Entity(BaseModel):
    entity_type: EntityType
    surface_form: str
    span: EntitySpan


class NERMetadata(BaseModel):
    model_name: str
    entity_count: int
    processing_time_ms: float


class NERResponse(BaseModel):
    entities: List[Entity]
    metadata: NERMetadata

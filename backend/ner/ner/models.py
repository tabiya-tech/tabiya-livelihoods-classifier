"""Pydantic request/response models for the NER API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class NERRequest(BaseModel):
    text: str = Field(..., description="Job-related text to extract entities from")
    entity_types: Optional[List[str]] = Field(
        None, description="Filter to specific entity types (e.g. ['occupation', 'skill'])"
    )


class EntitySpan(BaseModel):
    start: int
    end: int


class Entity(BaseModel):
    entity_type: str
    surface_form: str
    span: EntitySpan


class NERMetadata(BaseModel):
    model_name: str
    entity_count: int
    processing_time_ms: float


class NERResponse(BaseModel):
    entities: List[Entity]
    metadata: NERMetadata

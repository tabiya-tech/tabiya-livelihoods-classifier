"""Pydantic request/response models for the NEL API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class EntityInput(BaseModel):
    text: str = Field(..., min_length=1)
    entity_type: str


class NELOptions(BaseModel):
    top_k: Optional[int] = Field(5, ge=1, le=50)
    min_similarity: Optional[float] = Field(0.0, ge=0.0, le=1.0)


class NELRequest(BaseModel):
    entities: List[EntityInput] = Field(..., min_length=1)
    options: Optional[NELOptions] = None


class TaxonomyMatch(BaseModel):
    similarity_score: float
    taxonomy: str
    label: str
    code: Optional[str] = None
    uri: Optional[str] = None
    eqf_level: Optional[str] = None


class LinkedEntity(BaseModel):
    input_text: str
    entity_type: str
    matches: List[TaxonomyMatch]


class NELMetadata(BaseModel):
    linker_model: str
    taxonomy: str
    processing_time_ms: float


class NELResponse(BaseModel):
    linked_entities: List[LinkedEntity]
    metadata: NELMetadata

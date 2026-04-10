"""Pydantic request/response models for the Classify API."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ClassifyOptions(BaseModel):
    extract_entities: Optional[List[str]] = None
    top_k: int = Field(5, ge=1, le=50)
    min_similarity: float = Field(0.0, ge=0.0, le=1.0)


class ClassifyRequest(BaseModel):
    text: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    options: Optional[ClassifyOptions] = None


class BatchJob(BaseModel):
    job_id: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None


class BatchRequest(BaseModel):
    jobs: List[BatchJob] = Field(..., min_length=1)
    options: Optional[ClassifyOptions] = None


class EntitySpan(BaseModel):
    start: int
    end: int


class ClassifiedEntity(BaseModel):
    entity_type: str
    surface_form: str
    span: EntitySpan
    linked_entities: Optional[List[dict]] = None


class Classification(BaseModel):
    entities: List[ClassifiedEntity]
    entity_counts: Dict[str, int]


class ClassifyMetadata(BaseModel):
    classifier_version: str
    model_name: str
    linker_model: str
    processing_time_ms: float
    input_text_hash: str


class ClassifyResponse(BaseModel):
    classification: Classification
    metadata: ClassifyMetadata

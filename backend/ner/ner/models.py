"""Pydantic request/response models for the NER API."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# Canonical NER labels for tabiya/roberta-base-job-ner: see training/training/train_ner.py LABEL_LIST
# (occupation, skill, qualification, experience, domain).


class NERRequest(BaseModel):
    text: str = Field(..., description="Job-related text to extract entities from.")
    entity_types: Optional[List[str]] = Field(
        None,
        description=(
            "Filter results to specific entity types (lowercase). "
            "Omit to return all types. "
            "Typical values: occupation, skill, qualification, experience, domain."
        ),
    )

    @field_validator("entity_types", mode="before")
    @classmethod
    def _normalize_entity_types(cls, v):
        if v is None:
            return None
        if not isinstance(v, list):
            return v
        return [str(x).strip().lower() for x in v if str(x).strip()]


class EntitySpan(BaseModel):
    start: int
    end: int


class Entity(BaseModel):
    """NER span; ``entity_type`` is a string so new model labels do not break the API."""

    entity_type: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description=(
            "Lowercase type from the model (e.g. occupation, skill, qualification, experience, domain)."
        ),
    )
    surface_form: str
    span: EntitySpan

    @field_validator("entity_type", mode="before")
    @classmethod
    def _lower_entity_type(cls, v: object) -> str:
        return str(v).strip().lower()


class NERMetadata(BaseModel):
    model_name: str
    entity_count: int
    processing_time_ms: float


class NERResponse(BaseModel):
    entities: List[Entity]
    metadata: NERMetadata

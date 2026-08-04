"""Pydantic request/response models for the NEL API."""

from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    occupation = "occupation"
    skill = "skill"
    qualification = "qualification"


class EntityInput(BaseModel):
    text: str = Field(..., min_length=1, description="Surface form of the entity to link.")
    entity_type: EntityType = Field(..., description="Type of the entity. Allowed values: 'occupation', 'skill', 'qualification'.")


class NELOptions(BaseModel):
    top_k: Optional[int] = Field(5, ge=1, le=50, description="Maximum number of ESCO matches to return per entity (1–50).")
    min_similarity: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Minimum cosine similarity score for a match to be included (0.0–1.0).")


class NELRequest(BaseModel):
    entities: List[EntityInput] = Field(..., min_length=1, description="List of entities to link. Minimum 1 item.")
    options: Optional[NELOptions] = None
    language: Optional[str] = Field(
        None,
        description=(
            "Language of the entity text, selecting the taxonomy label pack to link against. "
            "Accepts a language code ('en', 'es') or a taxonomy locale ('AR-es'). "
            "Omit to use the service default (TARGET_LANGUAGE, else 'en')."
        ),
        examples=["es"],
    )


class TaxonomyMatch(BaseModel):
    similarity_score: float = Field(..., description="Cosine similarity between the entity and the ESCO entry (0.0–1.0).")
    taxonomy: Literal["esco"] = Field(..., description="Taxonomy source. Always 'esco'.")
    label: str = Field(..., description="Human-readable ESCO label.")
    code: Optional[str] = Field(None, description="ESCO occupation code (occupations only).")
    uri: Optional[str] = Field(None, description="ESCO URI for the matched entry.")
    eqf_level: Optional[str] = Field(None, description="European Qualifications Framework level (qualifications only).")


class LinkedEntity(BaseModel):
    input_text: str
    entity_type: EntityType
    matches: List[TaxonomyMatch] = Field(..., description="ESCO matches ordered by similarity score descending.")


class NELMetadata(BaseModel):
    linker_model: str = Field(..., description="Sentence-transformer model used for similarity scoring.")
    taxonomy: Literal["esco"] = Field(..., description="Taxonomy used for linking. Always 'esco'.")
    processing_time_ms: float
    language: Optional[str] = Field(None, description="Language the entities were linked in.")
    taxonomy_locale: Optional[str] = Field(
        None, description="Locale of the taxonomy label pack used (e.g. 'en', 'AR-es')."
    )


class NELResponse(BaseModel):
    linked_entities: List[LinkedEntity]
    metadata: NELMetadata

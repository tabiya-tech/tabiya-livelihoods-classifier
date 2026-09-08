"""Pydantic request/response models for the NER API."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Entity types accepted as REQUEST filters.

    The model can also emit `experience` and `domain` tags, which are not
    linkable against ESCO and so make poor request-time filters; we keep
    the request enum narrow but allow the response to surface any string
    the model produces (see `Entity` below).
    """

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
    language: Optional[str] = Field(
        None,
        description=(
            "Language of the text, selecting which extraction model runs. "
            "Accepts a language code ('en', 'es') or a locale ('AR-es'). "
            "Omit to use the service default (TARGET_LANGUAGE, else 'en')."
        ),
        examples=["es"],
    )


class EntitySpan(BaseModel):
    start: int
    end: int


class Entity(BaseModel):
    # The model emits more types than the request-side enum advertises
    # (e.g. "experience", "domain"). We surface them verbatim as strings so
    # the response is never silently rejected by Pydantic.
    entity_type: str
    surface_form: str
    span: EntitySpan


class NERMetadata(BaseModel):
    model_name: str
    entity_count: int
    processing_time_ms: float
    language: Optional[str] = Field(None, description="Language the text was processed as.")
    model_is_language_specific: Optional[bool] = Field(
        None,
        description=(
            "False when this language has no checkpoint of its own and was served by "
            "another language's model — expect degraded extraction."
        ),
    )


class NERResponse(BaseModel):
    entities: List[Entity]
    metadata: NERMetadata

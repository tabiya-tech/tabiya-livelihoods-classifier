"""Pydantic request/response models for the Classify API."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Aligned with tabiya/roberta-base-job-ner labels (training/training/train_ner.py)."""

    occupation = "occupation"
    skill = "skill"
    qualification = "qualification"
    experience = "experience"
    domain = "domain"


class BatchStatus(str, Enum):
    processing = "processing"
    completed = "completed"
    failed = "failed"


class JobStatus(str, Enum):
    completed = "completed"
    error = "error"


class ClassifyOptions(BaseModel):
    extract_entities: Optional[List[EntityType]] = Field(
        None,
        description=(
            "Restrict extraction to specific entity types. "
            "Omit to extract all types. "
            "Values: occupation, skill, qualification, experience, domain."
        ),
    )
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Maximum number of ESCO matches to return per entity (1–50).",
    )
    min_similarity: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score for a match to be included (0.0–1.0).",
    )


class ClassifyRequest(BaseModel):
    text: Optional[str] = Field(None, description="Raw job ad text. Use this OR title + description.")
    title: Optional[str] = Field(None, description="Job title. Combined with description when text is omitted.")
    description: Optional[str] = Field(None, description="Job description body. Combined with title when text is omitted.")
    options: Optional[ClassifyOptions] = None


class BatchJob(BaseModel):
    job_id: Optional[str] = Field(None, description="Caller-supplied identifier returned verbatim in results. Defaults to 'job_0', 'job_1', … if omitted.")
    text: Optional[str] = Field(None, description="Raw job ad text. Use this OR title + description.")
    title: Optional[str] = Field(None, description="Job title.")
    description: Optional[str] = Field(None, description="Job description body.")


class BatchRequest(BaseModel):
    jobs: List[BatchJob] = Field(..., min_length=1, description="List of jobs to classify. Minimum 1 item.")
    options: Optional[ClassifyOptions] = Field(None, description="Classification options applied uniformly to all jobs.")


class EntitySpan(BaseModel):
    start: int = Field(..., description="Character offset of the start of the surface form in the input text.")
    end: int = Field(..., description="Character offset of the end of the surface form in the input text.")


class ClassifiedEntity(BaseModel):
    entity_type: str = Field(..., description="Type of the extracted entity.")
    surface_form: str = Field(..., description="The exact text span extracted from the input.")
    span: EntitySpan
    linked_entities: Optional[List[dict]] = Field(None, description="ESCO matches for this entity, ordered by similarity score descending.")


class Classification(BaseModel):
    entities: List[ClassifiedEntity]
    entity_counts: Dict[str, int] = Field(..., description="Number of entities extracted per type.")


class ClassifyMetadata(BaseModel):
    classifier_version: str
    model_name: str = Field(..., description="HuggingFace model used for NER.")
    linker_model: str = Field(..., description="Sentence-transformer model used for NEL similarity.")
    processing_time_ms: float
    input_text_hash: str = Field(..., description="SHA-256 hash of the input text, for deduplication.")


class ClassifyResponse(BaseModel):
    classification: Classification
    metadata: ClassifyMetadata


# ── Batch response models ──────────────────────────────────────────────────

class BatchSubmitResponse(BaseModel):
    batch_id: str
    status: BatchStatus
    total: int = Field(..., description="Total number of jobs submitted.")


class BatchStatusResponse(BaseModel):
    batch_id: str
    status: BatchStatus
    total: int
    processed: int = Field(..., description="Number of jobs processed so far.")


class BatchJobResult(BaseModel):
    job_id: str
    status: JobStatus
    error: Optional[str] = Field(None, description="Error message when status is 'error'.")
    classification: Optional[Classification] = None
    metadata: Optional[ClassifyMetadata] = None


class BatchResultsResponse(BaseModel):
    batch_id: str
    status: BatchStatus
    total: int
    processed: int
    results: List[BatchJobResult]

"""Persisted shape of a classification record.

Thin record — no input text, no entities. Enough for usage stats and a
history list. See classifier-redesign-plan.md Step 9 for the known gap
around full-result storage and the foreign key that would link here.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ClassificationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification_id: str
    user_id: str
    pipeline_id: str
    entity_count: int
    processing_time_ms: float
    created_at: datetime

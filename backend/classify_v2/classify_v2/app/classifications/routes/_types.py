"""Request/response types for classification history and usage endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ClassificationSummary(BaseModel):
    classification_id: str
    pipeline_id: str
    entity_count: int
    processing_time_ms: float
    created_at: datetime


class ClassificationsPage(BaseModel):
    items: list[ClassificationSummary]
    next_cursor: Optional[str] = None


class DailyCount(BaseModel):
    date: str = Field(description="Calendar date in YYYY-MM-DD format (UTC).")
    count: int


class UsageResponse(BaseModel):
    days: int
    data: list[DailyCount]

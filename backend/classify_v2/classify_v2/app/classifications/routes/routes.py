"""Classification history and usage endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from classify_v2.app.auth.firebase import get_firebase_uid
from classify_v2.app.classifications.repository import (
    ClassificationRepository,
    IClassificationRepository,
)
from classify_v2.app.classifications.routes._types import (
    ClassificationSummary,
    ClassificationsPage,
    DailyCount,
    UsageResponse,
)
from classify_v2.app.server_dependencies.db_dependencies import ClassifyDBProvider

_logger = logging.getLogger(__name__)

router = APIRouter(tags=["classifications"])


async def _get_classifications_repo() -> IClassificationRepository:
    app_db = await ClassifyDBProvider.get_application_db()
    return ClassificationRepository(app_db)


@router.get("/v2/classifications", response_model=ClassificationsPage)
async def list_classifications(
    limit: int = Query(20, ge=1, le=100),
    cursor: str | None = Query(None),
    uid: str = Depends(get_firebase_uid),
    repo: IClassificationRepository = Depends(_get_classifications_repo),
) -> ClassificationsPage:
    records, next_cursor = await repo.list_for_user(
        uid, limit=limit, cursor=cursor
    )
    return ClassificationsPage(
        items=[
            ClassificationSummary(
                classification_id=record.classification_id,
                pipeline_id=record.pipeline_id,
                entity_count=record.entity_count,
                processing_time_ms=record.processing_time_ms,
                created_at=record.created_at,
            )
            for record in records
        ],
        next_cursor=next_cursor,
    )


@router.get("/v2/usage", response_model=UsageResponse)
async def get_usage(
    days: int = Query(30, ge=1, le=365),
    uid: str = Depends(get_firebase_uid),
    repo: IClassificationRepository = Depends(_get_classifications_repo),
) -> UsageResponse:
    raw = await repo.daily_counts_for_user(uid, days=days)
    return UsageResponse(
        days=days,
        data=[DailyCount(date=entry["date"], count=entry["count"]) for entry in raw],
    )

"""Classify v2 routes."""

from __future__ import annotations

import logging
import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from classify_v2.app.auth.firebase import get_firebase_uid
from classify_v2.app.classification.service.errors import (
    EmbeddingsCacheNotReadyError,
    NELServiceError,
    NERServiceError,
)
from classify_v2.app.classification.service.service import (
    ClassifyService,
    IClassifyService,
)
from classify_v2.app.classification.service.types import (
    ClassifyRequest,
    ClassifyResponse,
)
from classify_v2.app.pipelines.executor import PipelineExecutor
from classify_v2.app.pipelines.plugins_routes.routes import (
    get_plugin_http,
    get_plugin_registry,
)
from classify_v2.app.pipelines.registry import PluginRegistry
from classify_v2.app.pipelines.repository import (
    PipelineDocument,
    PipelineNotFoundError,
    PipelineRepository,
)
from classify_v2.app.pipelines.service import (
    IPipelineService,
    PipelineService,
    PipelineValidator,
)
from classify_v2.app.pipelines.service.service import DefaultTabiyaConfig
from classify_v2.app.server_dependencies.db_dependencies import ClassifyDBProvider
from classify_v2.config import MAX_TEXT_LENGTH

_logger = logging.getLogger(__name__)

# Dedicated summary logger — design §11 asks for one top-line log per
# classify call in addition to the per-stage lines the executor emits.
# Consumers filter on `logger.name == "classify_v2.classify_summary"`.
_summary_logger = logging.getLogger("classify_v2.classify_summary")

router = APIRouter(tags=["classify"])


async def _get_pipeline_service(
    registry: PluginRegistry = Depends(get_plugin_registry),
) -> IPipelineService:
    app_db = await ClassifyDBProvider.get_application_db()
    repository = PipelineRepository(app_db)
    validator = PipelineValidator(registry)
    return PipelineService(
        repository=repository, validator=validator, registry=registry
    )


def _get_classify_service(request: Request) -> IClassifyService:
    registry: PluginRegistry = request.app.state.plugin_registry
    http_client = request.app.state.plugin_http
    executor = PipelineExecutor(registry=registry, http_client=http_client)
    return ClassifyService(executor=executor)


def _default_tabiya_config() -> DefaultTabiyaConfig | None:
    nel_model_id = os.getenv("DEFAULT_NEL_MODEL_ID", "")
    taxonomy_model_id = os.getenv("DEFAULT_TAXONOMY_MODEL_ID", "")
    if not nel_model_id or not taxonomy_model_id:
        return None
    return DefaultTabiyaConfig(
        nel_model_id=nel_model_id, taxonomy_model_id=taxonomy_model_id
    )


async def _resolve_pipeline(
    *,
    uid: str,
    explicit_pipeline_id: str | None,
    pipelines: IPipelineService,
) -> PipelineDocument:
    """Resolution order: explicit id → user's active pipeline → seeded default.

    Raises HTTPException(404) when an explicit id is passed but doesn't
    exist; raises HTTPException(400) when no pipeline can be resolved
    (seeding env vars unset AND user has none).
    """

    if explicit_pipeline_id:
        try:
            return await pipelines.get(user_id=uid, pipeline_id=explicit_pipeline_id)
        except PipelineNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    docs = await pipelines.list_for_user(uid)
    for doc in docs:
        if doc.is_active:
            return doc

    default_config = _default_tabiya_config()
    if default_config is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No active pipeline for this user. Create and activate one via "
                "/v2/pipelines, or ask an admin to configure "
                "DEFAULT_NEL_MODEL_ID + DEFAULT_TAXONOMY_MODEL_ID to enable "
                "automatic Default Tabiya seeding."
            ),
        )
    return await pipelines.ensure_default(user_id=uid, default_config=default_config)


def _build_input_text(req: ClassifyRequest) -> str:
    if req.text:
        return req.text.strip()
    parts = [req.title or "", req.description or ""]
    return "\n".join(p.strip() for p in parts if p.strip())


@router.post("/v2/classify", response_model=ClassifyResponse)
async def classify(
    request: ClassifyRequest,
    uid: str = Depends(get_firebase_uid),
    pipelines: IPipelineService = Depends(_get_pipeline_service),
    svc: IClassifyService = Depends(_get_classify_service),
):
    input_text = _build_input_text(request)
    if not input_text:
        raise HTTPException(
            status_code=400,
            detail="Provide 'text' or 'title'+'description'",
        )
    if len(input_text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Text exceeds maximum length ({MAX_TEXT_LENGTH} chars)",
        )

    pipeline = await _resolve_pipeline(
        uid=uid,
        explicit_pipeline_id=request.pipeline_id,
        pipelines=pipelines,
    )
    request_id = str(uuid.uuid4())
    _logger.info(
        "Classify v2 request %s: %d chars, pipeline=%s",
        request_id,
        len(input_text),
        pipeline.pipeline_id,
    )
    try:
        result = await svc.classify(
            pipeline=pipeline,
            input_text=input_text,
            options=request.options,
            request_id=request_id,
            user_id=uid,
        )
    except EmbeddingsCacheNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except NELServiceError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except NERServiceError as exc:
        _logger.error("Classify v2 failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))

    _logger.info(
        "Classify v2 done %s: %d entities in %.1fms",
        request_id,
        len(result.entities),
        result.metadata.processing_time_ms,
    )
    _summary_logger.info(
        "classify %s pipeline=%s stages=%d %.1fms",
        request_id,
        pipeline.pipeline_id,
        len(pipeline.stages),
        result.metadata.processing_time_ms,
        extra={
            "request_id": request_id,
            "pipeline_id": pipeline.pipeline_id,
            "pipeline_name": pipeline.name,
            "total_stages": len(pipeline.stages),
            "total_duration_ms": round(result.metadata.processing_time_ms, 3),
            "entity_count": len(result.entities),
            "user_id": uid,
        },
    )
    return result

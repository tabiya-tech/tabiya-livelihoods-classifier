"""Pipelines CRUD routes.

Endpoints:
  GET    /v2/pipelines                           — list for caller (seeds Default lazily)
  POST   /v2/pipelines                           — create
  POST   /v2/pipelines/validate                  — validate without persisting
  GET    /v2/pipelines/{pipeline_id}             — read one
  PUT    /v2/pipelines/{pipeline_id}             — update
  DELETE /v2/pipelines/{pipeline_id}             — delete
  POST   /v2/pipelines/{pipeline_id}/activate    — mark as caller's active pipeline
  POST   /v2/pipelines/{pipeline_id}/clone       — clone as an editable copy

All routes require an authenticated user via `get_firebase_uid`.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from classify_v2.app.auth.firebase import get_firebase_uid
from classify_v2.app.pipelines.plugins_routes.routes import get_plugin_registry
from classify_v2.app.pipelines.registry import PluginRegistry
from classify_v2.app.pipelines.repository import (
    PipelineDocument,
    PipelineNotFoundError,
    PipelineRepository,
    ReadonlyPipelineError,
)
from classify_v2.app.pipelines.routes._types import (
    ListPipelinesResponse,
    ValidatePipelineRequest,
    ValidatePipelineResponse,
)
from classify_v2.app.pipelines.service import (
    CreatePipelineInput,
    IPipelineService,
    PipelineService,
    PipelineValidationError,
    PipelineValidator,
    UpdatePipelineInput,
)
from classify_v2.app.pipelines.service.service import DefaultTabiyaConfig
from classify_v2.app.server_dependencies.db_dependencies import ClassifyDBProvider
import os

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/pipelines", tags=["pipelines"])


async def get_pipeline_service(
    registry: PluginRegistry = Depends(get_plugin_registry),
) -> IPipelineService:
    """Build the service on demand from app.state.plugin_registry + Mongo."""

    app_db = await ClassifyDBProvider.get_application_db()
    repository = PipelineRepository(app_db)
    validator = PipelineValidator(registry)
    return PipelineService(
        repository=repository, validator=validator, registry=registry
    )


def _default_tabiya_config() -> DefaultTabiyaConfig | None:
    """Read env vars at call time so tests can toggle them per case.

    Returns `None` when either default is unset — the caller then skips
    seeding, which is the intended behaviour for a deployment that hasn't
    configured seeding yet.
    """

    nel_model_id = os.getenv("DEFAULT_NEL_MODEL_ID", "")
    taxonomy_model_id = os.getenv("DEFAULT_TAXONOMY_MODEL_ID", "")
    if not nel_model_id or not taxonomy_model_id:
        return None
    return DefaultTabiyaConfig(
        nel_model_id=nel_model_id,
        taxonomy_model_id=taxonomy_model_id,
    )


def _validation_error_response(exc: PipelineValidationError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            "message": "Pipeline validation failed.",
            "issues": [issue.model_dump() for issue in exc.issues],
        },
    )


@router.get("", response_model=ListPipelinesResponse)
async def list_pipelines(
    uid: str = Depends(get_firebase_uid),
    svc: IPipelineService = Depends(get_pipeline_service),
) -> ListPipelinesResponse:
    # Lazy seed the default so API-only callers don't need the UI to bootstrap.
    default_config = _default_tabiya_config()
    if default_config is not None:
        try:
            await svc.ensure_default(user_id=uid, default_config=default_config)
        except Exception:
            # Never fail a list request because seeding hiccuped — surface
            # whatever pipelines the user does have and log the seed error.
            _logger.exception("ensure_default failed for user=%s", uid)
    pipelines = await svc.list_for_user(uid)
    return ListPipelinesResponse(pipelines=pipelines)


@router.post("", status_code=status.HTTP_201_CREATED, response_model=PipelineDocument)
async def create_pipeline(
    request: CreatePipelineInput,
    uid: str = Depends(get_firebase_uid),
    svc: IPipelineService = Depends(get_pipeline_service),
) -> PipelineDocument:
    try:
        return await svc.create(user_id=uid, request=request)
    except PipelineValidationError as exc:
        raise _validation_error_response(exc)


@router.post("/validate", response_model=ValidatePipelineResponse)
async def validate_pipeline(
    request: ValidatePipelineRequest,
    _uid: str = Depends(get_firebase_uid),
    svc: IPipelineService = Depends(get_pipeline_service),
) -> ValidatePipelineResponse:
    issues = await svc.validate(request.stages)
    return ValidatePipelineResponse(valid=not issues, issues=issues)


@router.get("/{pipeline_id}", response_model=PipelineDocument)
async def get_pipeline(
    pipeline_id: str,
    uid: str = Depends(get_firebase_uid),
    svc: IPipelineService = Depends(get_pipeline_service),
) -> PipelineDocument:
    try:
        return await svc.get(user_id=uid, pipeline_id=pipeline_id)
    except PipelineNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.put("/{pipeline_id}", response_model=PipelineDocument)
async def update_pipeline(
    pipeline_id: str,
    request: UpdatePipelineInput,
    uid: str = Depends(get_firebase_uid),
    svc: IPipelineService = Depends(get_pipeline_service),
) -> PipelineDocument:
    try:
        return await svc.update(
            user_id=uid, pipeline_id=pipeline_id, request=request
        )
    except PipelineNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ReadonlyPipelineError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except PipelineValidationError as exc:
        raise _validation_error_response(exc)


@router.delete(
    "/{pipeline_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_pipeline(
    pipeline_id: str,
    uid: str = Depends(get_firebase_uid),
    svc: IPipelineService = Depends(get_pipeline_service),
) -> Response:
    try:
        await svc.delete(user_id=uid, pipeline_id=pipeline_id)
    except PipelineNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ReadonlyPipelineError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{pipeline_id}/activate", response_model=PipelineDocument)
async def activate_pipeline(
    pipeline_id: str,
    uid: str = Depends(get_firebase_uid),
    svc: IPipelineService = Depends(get_pipeline_service),
) -> PipelineDocument:
    try:
        return await svc.activate(user_id=uid, pipeline_id=pipeline_id)
    except PipelineNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except PipelineValidationError as exc:
        raise _validation_error_response(exc)


@router.post("/{pipeline_id}/clone", status_code=status.HTTP_201_CREATED, response_model=PipelineDocument)
async def clone_pipeline(
    pipeline_id: str,
    uid: str = Depends(get_firebase_uid),
    svc: IPipelineService = Depends(get_pipeline_service),
) -> PipelineDocument:
    try:
        return await svc.clone(user_id=uid, pipeline_id=pipeline_id)
    except PipelineNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

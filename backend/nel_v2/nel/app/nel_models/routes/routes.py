"""NEL models routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from nel.app.nel_models.repository.repository import NELModelRepository
from nel.app.nel_models.routes._types import NELModelResponse
from nel.app.nel_models.service.errors import NELModelNotFoundError
from nel.app.nel_models.service.service import INELModelService, NELModelService
from nel.app.server_dependencies.db_dependencies import ClassifierDBProvider

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/nel/models", tags=["nel-models"])


async def _get_service() -> INELModelService:
    app_db = await ClassifierDBProvider.get_application_db()
    return NELModelService(NELModelRepository(app_db))


@router.get("", response_model=list[NELModelResponse])
async def list_nel_models(svc: INELModelService = Depends(_get_service)):
    return await svc.get_all()


@router.get("/{model_id}", response_model=NELModelResponse)
async def get_nel_model(model_id: str, svc: INELModelService = Depends(_get_service)):
    try:
        return await svc.get(model_id)
    except NELModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"NEL model {model_id!r} not found")

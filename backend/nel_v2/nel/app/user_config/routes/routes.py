"""User config routes."""

import logging

from fastapi import APIRouter, Depends

from nel.app.user_config.repository.repository import UserConfigRepository
from nel.app.user_config.routes._types import UserConfigRequest, UserConfigResponse
from nel.app.user_config.routes.auth import get_firebase_uid
from nel.app.user_config.service.service import IUserConfigService, UserConfigService
from nel.app.user_config.service.types import UserConfig
from nel.app.server_dependencies.db_dependencies import ClassifierDBProvider

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/nel/user", tags=["user-config"])


async def _get_service() -> IUserConfigService:
    app_db = await ClassifierDBProvider.get_application_db()
    return UserConfigService(UserConfigRepository(app_db))


@router.get("/config", response_model=UserConfigResponse)
async def get_user_config(
    uid: str = Depends(get_firebase_uid),
    svc: IUserConfigService = Depends(_get_service),
):
    config = await svc.get(uid)
    return UserConfigResponse(
        taxonomy_model_id=config.taxonomy_model_id,
        nel_model_id=config.nel_model_id,
    )


@router.put("/config", response_model=UserConfigResponse)
async def update_user_config(
    request: UserConfigRequest,
    uid: str = Depends(get_firebase_uid),
    svc: IUserConfigService = Depends(_get_service),
):
    config = await svc.upsert(UserConfig(
        user_id=uid,
        taxonomy_model_id=request.taxonomy_model_id,
        nel_model_id=request.nel_model_id,
    ))
    return UserConfigResponse(
        taxonomy_model_id=config.taxonomy_model_id,
        nel_model_id=config.nel_model_id,
    )

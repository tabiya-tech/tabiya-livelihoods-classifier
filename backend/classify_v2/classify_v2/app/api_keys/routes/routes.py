"""api-keys routes for classify-v2.

GET    /v2/user/api-keys           — list active keys for the caller
POST   /v2/user/api-keys           — provision a new key (label in body)
DELETE /v2/user/api-keys/{key_id}  — revoke a key

All three require an authenticated Firebase user (or local-mode bypass).
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from classify_v2.app.api_keys.repository.repository import ApiKeysRepository
from classify_v2.app.api_keys.routes._types import (
    CreateApiKeyRequest,
    CreateApiKeyResponse,
    ListApiKeysResponse,
)
from classify_v2.app.api_keys.service.errors import (
    ApiKeyNotFoundError,
    ApiKeysQuotaExceededError,
    GcpApiKeysError,
)
from classify_v2.app.api_keys.service.gcp_key_manager import IGcpKeyManager
from classify_v2.app.api_keys.service.service import ApiKeysService, IApiKeysService
from classify_v2.app.auth.firebase import get_firebase_uid
from classify_v2.app.server_dependencies.db_dependencies import ClassifyDBProvider
from classify_v2.config import MAX_API_KEYS_PER_USER

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/user/api-keys", tags=["api-keys"])


# ── Dependency wiring ─────────────────────────────────────────────────────
#
# The GCP API Keys async client must be constructed on the main event loop
# (uvloop refuses asyncio.get_event_loop() from worker threads). main.py's
# lifespan creates the singleton and stores it on app.state; we read it here.


def _get_gcp_manager(request: Request) -> IGcpKeyManager:
    gcp = getattr(request.app.state, "gcp_key_manager", None)
    if gcp is None:
        raise RuntimeError(
            "GCP key manager was not initialised during app startup; "
            "check classify_v2.main.lifespan",
        )
    return gcp


async def _get_service(
    gcp: IGcpKeyManager = Depends(_get_gcp_manager),
) -> IApiKeysService:
    app_db = await ClassifyDBProvider.get_application_db()
    return ApiKeysService(
        repository=ApiKeysRepository(app_db),
        gcp=gcp,
        max_keys_per_user=MAX_API_KEYS_PER_USER,
    )


# ── Routes ────────────────────────────────────────────────────────────────


@router.get("", response_model=ListApiKeysResponse)
async def list_api_keys(
    uid: str = Depends(get_firebase_uid),
    svc: IApiKeysService = Depends(_get_service),
) -> ListApiKeysResponse:
    keys = await svc.list_keys(user_id=uid)
    return ListApiKeysResponse(keys=keys)


@router.post("", status_code=status.HTTP_201_CREATED, response_model=CreateApiKeyResponse)
async def create_api_key(
    request: CreateApiKeyRequest,
    uid: str = Depends(get_firebase_uid),
    svc: IApiKeysService = Depends(_get_service),
) -> CreateApiKeyResponse:
    try:
        issued = await svc.create_key(user_id=uid, label=request.label)
    except ApiKeysQuotaExceededError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except GcpApiKeysError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    return CreateApiKeyResponse(key=issued.key, meta=issued.meta)


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: str,
    uid: str = Depends(get_firebase_uid),
    svc: IApiKeysService = Depends(_get_service),
) -> None:
    try:
        await svc.revoke_key(user_id=uid, key_id=key_id)
    except ApiKeyNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key not found")
    except GcpApiKeysError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

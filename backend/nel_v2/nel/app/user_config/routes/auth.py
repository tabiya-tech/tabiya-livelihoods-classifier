"""Firebase auth dependency for nel-v2.

Three auth paths:
  - Production (Firebase): API Gateway verifies the Firebase JWT and forwards
    decoded user info as base64 JSON in the x-apigateway-api-userinfo header.
  - Production (API key):  API Gateway verifies the API key but does not set
    x-apigateway-api-userinfo. We recover the owning user by hashing the
    forwarded `x-api-key` and looking it up in the api_keys collection, so an
    API-key caller resolves to the same uid (and per-user model config) as
    that user's Firebase session. Unresolvable keys fall back to a fixed
    service UID (default config), preserving prior behaviour.
  - Local:      Auth is skipped entirely — a fixed uid is returned so the service
    is usable without any auth setup.
"""

import base64
import json
import logging

from fastapi import HTTPException, Request, status

from nel.app.server_dependencies.db_dependencies import ClassifierDBProvider
from nel.app.user_config.routes.api_key_resolver import resolve_user_id_from_api_key
from nel.config import TARGET_ENVIRONMENT_TYPE

_logger = logging.getLogger(__name__)

_LOCAL_UID = "local-user"
_API_KEY_UID = "api-key-user"

_API_KEY_HEADER = "x-api-key"


def _decode_gateway_user_info(auth_info_b64: str) -> dict:
    padding_needed = len(auth_info_b64) % 4
    if padding_needed == 1:
        raise ValueError("Invalid base64 input")
    elif padding_needed == 2:
        auth_info_b64 += "=="
    elif padding_needed == 3:
        auth_info_b64 += "="
    decoded = base64.b64decode(auth_info_b64.encode("utf-8"))
    return json.loads(decoded.decode("utf-8"))


async def get_firebase_uid(request: Request) -> str:
    """FastAPI dependency: returns the uid of the authenticated user.

    In local development auth is skipped and a fixed uid is returned.
    In production the API Gateway has already verified the request and placed
    either the decoded Firebase claims in x-apigateway-api-userinfo or a
    validated `x-api-key`.
    """
    if TARGET_ENVIRONMENT_TYPE == "local":
        return _LOCAL_UID

    try:
        auth_info_b64 = request.headers.get("x-apigateway-api-userinfo")
        if auth_info_b64:
            token_info = _decode_gateway_user_info(auth_info_b64)
            uid = token_info.get("sub") or token_info.get("user_id")
            if not uid:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
            return uid

        # No Firebase claims — the gateway authenticated this via API key.
        # Recover the owning user so their per-user config applies.
        return await _resolve_api_key_uid(request)
    except HTTPException:
        raise
    except Exception as exc:
        _logger.warning("Auth error: %s — %s", exc.__class__.__name__, exc)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


async def _resolve_api_key_uid(request: Request) -> str:
    """Map a validated `x-api-key` to its owner, or the shared service uid.

    A missing or unrecognised key falls back to the shared `_API_KEY_UID`
    rather than 401 — the gateway already gated access, so this only affects
    *which config* applies, and the shared uid preserves the pre-existing
    default-config behaviour for keys that predate ownership tracking.
    """
    plaintext_key = request.headers.get(_API_KEY_HEADER)
    if not plaintext_key:
        return _API_KEY_UID
    try:
        application_db = await ClassifierDBProvider.get_application_db()
        owner_uid = await resolve_user_id_from_api_key(application_db, plaintext_key)
    except Exception as exc:
        _logger.warning("API-key uid resolution failed: %s — %s", exc.__class__.__name__, exc)
        return _API_KEY_UID
    return owner_uid or _API_KEY_UID

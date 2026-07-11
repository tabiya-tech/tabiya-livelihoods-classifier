"""Firebase auth dependency for classify-v2.

Three auth paths (same as nel-v2):
  - local:  TARGET_ENVIRONMENT_TYPE=local → return a fixed uid, no token check.
  - Firebase via gateway: gateway has already verified the JWT and forwarded
    the decoded claims as base64 JSON in x-apigateway-api-userinfo.
  - API key via gateway:  gateway verified the key but forwards NO user-info
    header. We recover the owning user by hashing the forwarded `x-api-key`
    and looking it up in the api_keys collection — so an API-key caller
    resolves to the same uid (and per-user model config) as that user's
    Firebase session. Only if the key can't be resolved do we fall back to
    the shared service uid.
"""

import base64
import json
import logging
import os

from fastapi import HTTPException, Request, status

from classify_v2.app.auth.api_key_resolver import resolve_user_id_from_api_key
from classify_v2.app.server_dependencies.db_dependencies import ClassifyDBProvider

_logger = logging.getLogger(__name__)

_LOCAL_UID = "local-user"
_API_KEY_UID = "api-key-user"

_API_KEY_HEADER = "x-api-key"


def _decode_gateway_user_info(auth_info_b64: str) -> dict:
    padding_needed = len(auth_info_b64) % 4
    if padding_needed == 1:
        raise ValueError("Invalid base64 input")
    if padding_needed == 2:
        auth_info_b64 += "=="
    elif padding_needed == 3:
        auth_info_b64 += "="
    decoded = base64.b64decode(auth_info_b64.encode("utf-8"))
    return json.loads(decoded.decode("utf-8"))


async def get_firebase_uid(request: Request) -> str:
    """FastAPI dependency: returns the uid of the authenticated user.

    Local mode skips auth and returns a fixed uid. In production the gateway
    has already authenticated the request and forwarded either the decoded
    Firebase claims (x-apigateway-api-userinfo) or a validated `x-api-key`.

    Reads TARGET_ENVIRONMENT_TYPE live (not an import-time constant) so tests
    that set it via monkeypatch behave deterministically regardless of module
    import order.
    """
    if os.getenv("TARGET_ENVIRONMENT_TYPE", "") == "local":
        _logger.warning("AUTH BYPASS ACTIVE — TARGET_ENVIRONMENT_TYPE=local, returning fixed uid '%s'", _LOCAL_UID)
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
        application_db = await ClassifyDBProvider.get_application_db()
        owner_uid = await resolve_user_id_from_api_key(application_db, plaintext_key)
    except Exception as exc:
        _logger.warning("API-key uid resolution failed: %s — %s", exc.__class__.__name__, exc)
        return _API_KEY_UID
    return owner_uid or _API_KEY_UID

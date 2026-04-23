"""Firebase auth dependency for nel-v2.

Three auth paths:
  - Production (Firebase): API Gateway verifies the Firebase JWT and forwards
    decoded user info as base64 JSON in the x-apigateway-api-userinfo header.
  - Production (API key):  API Gateway verifies the API key but does not set
    x-apigateway-api-userinfo. A fixed service UID is used so the request is
    handled with the default service-level config.
  - Local:      Auth is skipped entirely — a fixed uid is returned so the service
    is usable without any auth setup.
"""

import base64
import json
import logging

from fastapi import HTTPException, Request, status

from nel.config import TARGET_ENVIRONMENT_TYPE

_logger = logging.getLogger(__name__)

_LOCAL_UID = "local-user"
_API_KEY_UID = "api-key-user"


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


def get_firebase_uid(request: Request) -> str:
    """FastAPI dependency: returns the Firebase uid of the authenticated user.

    In local development auth is skipped and a fixed uid is returned.
    In production the API Gateway has already verified the token and placed
    the decoded claims in x-apigateway-api-userinfo.
    """
    if TARGET_ENVIRONMENT_TYPE == "local":
        return _LOCAL_UID

    try:
        auth_info_b64 = request.headers.get("x-apigateway-api-userinfo")
        if not auth_info_b64:
            # No user info header — request was authenticated via API key.
            # The gateway already verified the key; use the shared service UID.
            return _API_KEY_UID
        token_info = _decode_gateway_user_info(auth_info_b64)
        uid = token_info.get("sub") or token_info.get("user_id")
        if not uid:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        return uid
    except HTTPException:
        raise
    except Exception as exc:
        _logger.warning("Auth error: %s — %s", exc.__class__.__name__, exc)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

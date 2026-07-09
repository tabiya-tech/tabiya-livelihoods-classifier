"""Firebase auth dependency for classify-v2.

Three auth paths (same as nel-v2):
  - local:  TARGET_ENVIRONMENT_TYPE=local → return a fixed uid, no token check.
  - Firebase via gateway: gateway has already verified the JWT and forwarded
    the decoded claims as base64 JSON in x-apigateway-api-userinfo.
  - API key via gateway:  gateway verified the key; no user-info header. We
    use a fixed service uid for "shared-service" requests.
"""

import base64
import json
import logging

from fastapi import HTTPException, Request, status

from classify_v2.config import TARGET_ENVIRONMENT_TYPE

_logger = logging.getLogger(__name__)

_LOCAL_UID = "local-user"
_API_KEY_UID = "api-key-user"


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


def get_firebase_uid(request: Request) -> str:
    """FastAPI dependency: returns the Firebase uid of the authenticated user.

    Local mode skips auth and returns a fixed uid. In production the gateway
    has already authenticated the request and forwarded the claims.
    """
    if TARGET_ENVIRONMENT_TYPE == "local":
        _logger.warning("AUTH BYPASS ACTIVE — TARGET_ENVIRONMENT_TYPE=local, returning fixed uid '%s'", _LOCAL_UID)
        return _LOCAL_UID

    try:
        auth_info_b64 = request.headers.get("x-apigateway-api-userinfo")
        if not auth_info_b64:
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

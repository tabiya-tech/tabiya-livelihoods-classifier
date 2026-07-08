"""GCP identity-token auth dependency for plugin bundles.

The orchestrator (classify_v2) mints an identity token via the GCP metadata
server and sends it as `Authorization: Bearer <token>`. Each bundle Cloud
Run service validates the token against its own service URL as the audience.

Local dev short-circuit: when `TARGET_ENVIRONMENT_TYPE=local`, no token is
required. This mirrors the pattern already used by the Firebase Auth
dependency in classify_v2.

Verifying the token requires `google-auth`, which we do not import at
module load time so the contracts package itself stays dependency-light.
The bundle's `pyproject.toml` declares `google-auth` as a dependency; the
adapter grabs it lazily on the first request.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import Header, HTTPException, status


def _is_local() -> bool:
    return os.getenv("TARGET_ENVIRONMENT_TYPE", "").lower() == "local"


async def require_identity_token(
    authorization: Optional[str] = Header(default=None),
) -> None:
    """FastAPI dependency: reject requests without a valid GCP identity token.

    No-op when `TARGET_ENVIRONMENT_TYPE=local`. In production, the audience
    is read from `PLUGIN_BUNDLE_AUDIENCE` (the bundle's own Cloud Run URL).
    """

    if _is_local():
        return

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )
    token = authorization[len("bearer ") :].strip()
    audience = os.getenv("PLUGIN_BUNDLE_AUDIENCE")
    if not audience:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PLUGIN_BUNDLE_AUDIENCE not configured.",
        )

    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"google-auth is required to verify identity tokens: {exc}",
        ) from exc

    try:
        id_token.verify_oauth2_token(token, google_requests.Request(), audience=audience)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid identity token: {exc}",
        ) from exc

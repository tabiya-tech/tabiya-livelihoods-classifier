"""GCP identity-token helper for the core bundle's outbound calls.

The core bundle proxies to the private `ner` and `nel` Cloud Run services.
Those services require a GCP identity token whose audience is the target
service URL. This mirrors the pattern already used by the v1 classify
orchestrator (`backend/classify/classify/get_classify_service.py`).

Returns `None` when running outside GCP (local dev / tests), so the callers
simply omit the `Authorization` header and reach the local services
unauthenticated.
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger("tabiya-core-bundle.identity")

_METADATA_TOKEN_URL = (
    "http://metadata.google.internal/computeMetadata/v1/"
    "instance/service-accounts/default/identity"
)


def fetch_identity_token(audience: str) -> str | None:
    """Fetch a GCP identity token for `audience` via the metadata server.

    Returns None when the metadata server is unreachable (e.g. local dev),
    in which case the caller proceeds without an Authorization header.
    """

    try:
        response = httpx.get(
            _METADATA_TOKEN_URL,
            params={"audience": audience},
            headers={"Metadata-Flavor": "Google"},
            timeout=2.0,
        )
        response.raise_for_status()
        return response.text
    except Exception:  # noqa: BLE001 — absence of a token is expected off-GCP
        return None


def bearer_headers(audience: str) -> dict[str, str]:
    """Return `{Authorization: Bearer <token>}` or an empty dict off-GCP."""

    token = fetch_identity_token(audience)
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}

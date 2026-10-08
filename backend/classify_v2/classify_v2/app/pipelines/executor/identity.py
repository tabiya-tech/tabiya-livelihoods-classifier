"""GCP identity-token provider for orchestrator → plugin-bundle auth.

The executor attaches an `Authorization: Bearer <token>` header to every
`/plugin/invoke` call so a private Cloud Run bundle will accept it. Two
gates must both be satisfied in production:

  1. Cloud Run IAM — only classify_v2's service account has `run.invoker`
     on the bundle service. This requires an identity token regardless of
     any app-level check.
  2. The bundle's app-level `require_identity_token` dependency, which
     validates the token's audience against the bundle's own URL.

Both are satisfied by the same token: a GCP-minted identity token whose
audience is the target bundle's base URL.

The executor depends on the `IIdentityTokenProvider` Protocol, never the
concrete class — tests pass a fake, local mode passes `None` (no header).
Minting uses `google.auth`, which is already available transitively via
`google-cloud-api-keys`; it is imported lazily so importing this module
never requires GCP libraries (keeps unit tests dependency-light).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Protocol
from urllib.parse import urlsplit

_logger = logging.getLogger(__name__)


class IIdentityTokenProvider(Protocol):
    """Async port: mint a GCP identity token for a target URL's audience."""

    async def get_id_token(self, url: str) -> Optional[str]: ...


def _audience_for(url: str) -> str:
    """Derive the token audience (scheme + host) from a plugin invoke URL.

    Cloud Run identity tokens are audience-scoped to the *service* URL, not
    the individual request path — so `https://bundle-abc.run.app/plugin/x/invoke`
    yields audience `https://bundle-abc.run.app`.
    """

    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


class GcpIdentityTokenProvider(IIdentityTokenProvider):
    """Real provider — mints identity tokens via the GCP metadata server.

    Tokens are cached per-audience by the underlying google-auth request
    object, which refreshes them shortly before expiry. Each call runs the
    blocking google-auth fetch in a worker thread so the event loop is not
    stalled.
    """

    async def get_id_token(self, url: str) -> Optional[str]:
        audience = _audience_for(url)
        try:
            return await asyncio.to_thread(self._fetch_id_token, audience)
        except Exception:  # noqa: BLE001 — auth failures must not crash the request path
            _logger.warning(
                "Failed to mint identity token for audience %s", audience, exc_info=True
            )
            return None

    @staticmethod
    def _fetch_id_token(audience: str) -> Optional[str]:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token

        request = google_requests.Request()
        return id_token.fetch_id_token(request, audience)

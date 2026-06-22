"""Thin async wrapper around google-cloud-api-keys.

Why this is its own file: the service layer depends on the IGcpKeyManager
Protocol, never the concrete Google client. Tests substitute a fake
implementation and verify behavior without hitting GCP. The real client is
only constructed in production wiring (the route dependency below).
"""

import logging
from dataclasses import dataclass
from typing import Protocol

from google.cloud import api_keys_v2
from google.cloud.api_keys_v2.types import ApiTarget, Key, Restrictions

from classify_v2.app.api_keys.service.errors import GcpApiKeysError

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IssuedGcpKey:
    """The pieces we need back from `create_key`: the GCP resource name (for
    later delete) and the plaintext key string (returned to the user once).
    """

    gcp_key_name: str
    key_id: str
    key_string: str


class IGcpKeyManager(Protocol):
    """Async port for GCP API Keys operations.

    Concrete implementations: GcpKeyManager (real client) and the in-memory
    fake used by tests. The service layer always depends on this protocol.
    """

    async def create_key(self, display_name: str) -> IssuedGcpKey: ...

    async def delete_key(self, gcp_key_name: str) -> None: ...


class GcpKeyManager(IGcpKeyManager):
    """Real adapter — calls google-cloud-api-keys."""

    def __init__(
        self,
        *,
        project_id: str,
        managed_service: str,
        location: str = "global",
        client: api_keys_v2.ApiKeysAsyncClient | None = None,
    ) -> None:
        self._project_id = project_id
        self._managed_service = managed_service
        self._location = location
        self._client = client or api_keys_v2.ApiKeysAsyncClient()

    async def create_key(self, display_name: str) -> IssuedGcpKey:
        key = Key(
            display_name=display_name,
            restrictions=Restrictions(
                api_targets=[ApiTarget(service=self._managed_service)],
            ),
        )
        parent = f"projects/{self._project_id}/locations/{self._location}"
        try:
            op = await self._client.create_key(parent=parent, key=key)
            created = await op.result()
        except Exception as exc:
            _logger.exception("GCP create_key failed")
            raise GcpApiKeysError(f"GCP create_key failed: {exc}") from exc

        gcp_key_id = created.name.split("/")[-1]
        return IssuedGcpKey(
            gcp_key_name=created.name,
            key_id=gcp_key_id,
            key_string=created.key_string,
        )

    async def delete_key(self, gcp_key_name: str) -> None:
        try:
            op = await self._client.delete_key(name=gcp_key_name)
            await op.result()
        except Exception as exc:
            _logger.warning("GCP delete_key failed for %s: %s", gcp_key_name, exc)
            raise GcpApiKeysError(f"GCP delete_key failed: {exc}") from exc

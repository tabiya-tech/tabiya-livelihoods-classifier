"""api-keys service — the single seam between routes and the world.

Responsibilities:
- enforce per-user key quota
- coordinate GCP key provisioning + Mongo persistence
- ensure the GCP key is cleaned up if Mongo insert fails (no orphans)
- delete from GCP first on revoke; Mongo soft-delete second so a successful
  GCP delete is never silently re-listed.
"""

import hashlib
import logging
import time
from abc import ABC, abstractmethod

from classify_v2.app.api_keys.repository.repository import IApiKeysRepository
from classify_v2.app.api_keys.service.errors import (
    ApiKeyNotFoundError,
    ApiKeysQuotaExceededError,
    GcpApiKeysError,
)
from classify_v2.app.api_keys.service.gcp_key_manager import IGcpKeyManager
from classify_v2.app.api_keys.service.types import ApiKeyMetadata, CreatedApiKey

_logger = logging.getLogger(__name__)


def _hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


class IApiKeysService(ABC):
    @abstractmethod
    async def list_keys(self, *, user_id: str) -> list[ApiKeyMetadata]: ...

    @abstractmethod
    async def create_key(self, *, user_id: str, label: str) -> CreatedApiKey: ...

    @abstractmethod
    async def revoke_key(self, *, user_id: str, key_id: str) -> None: ...


class ApiKeysService(IApiKeysService):
    def __init__(
        self,
        *,
        repository: IApiKeysRepository,
        gcp: IGcpKeyManager,
        max_keys_per_user: int,
    ) -> None:
        self._repo = repository
        self._gcp = gcp
        self._max_keys = max_keys_per_user

    async def list_keys(self, *, user_id: str) -> list[ApiKeyMetadata]:
        return await self._repo.list_active(user_id)

    async def create_key(self, *, user_id: str, label: str) -> CreatedApiKey:
        active_count = await self._repo.count_active(user_id)
        if active_count >= self._max_keys:
            raise ApiKeysQuotaExceededError(
                f"User {user_id} already holds {active_count}/{self._max_keys} active keys"
            )

        issued = await self._gcp.create_key(display_name=f"{label} ({user_id})")

        meta = ApiKeyMetadata(
            key_id=issued.key_id,
            user_id=user_id,
            label=label,
            created_at=time.time(),
            last_used_at=None,
            revoked=False,
        )
        try:
            await self._repo.insert(
                meta=meta,
                key_hash=_hash_key(issued.key_string),
                gcp_key_name=issued.gcp_key_name,
            )
        except Exception:
            # Mongo insert failed after GCP provisioning succeeded — roll the
            # GCP side back so we don't leak unowned billable keys. Best
            # effort: log and re-raise the original error.
            _logger.exception(
                "Mongo insert failed for newly-issued key %s; attempting GCP rollback",
                issued.gcp_key_name,
            )
            try:
                await self._gcp.delete_key(issued.gcp_key_name)
            except GcpApiKeysError:
                _logger.exception(
                    "Rollback delete also failed for %s — manual cleanup required",
                    issued.gcp_key_name,
                )
            raise

        return CreatedApiKey(key=issued.key_string, meta=meta)

    async def revoke_key(self, *, user_id: str, key_id: str) -> None:
        gcp_key_name = await self._repo.find_gcp_key_name(user_id=user_id, key_id=key_id)
        if gcp_key_name is None:
            raise ApiKeyNotFoundError(f"Key {key_id!r} not found for user {user_id}")

        try:
            await self._gcp.delete_key(gcp_key_name)
        except GcpApiKeysError:
            # If GCP delete fails we don't mark the row revoked — the next
            # call can retry. This keeps Mongo + GCP consistent.
            _logger.warning(
                "GCP delete failed for %s; Mongo not updated, retry allowed",
                gcp_key_name,
            )
            raise

        marked = await self._repo.mark_revoked(user_id=user_id, key_id=key_id)
        if not marked:
            # Concurrent revoke or row already gone — treat as success since
            # the GCP key is now deleted either way.
            _logger.info("mark_revoked was a no-op for %s/%s", user_id, key_id)

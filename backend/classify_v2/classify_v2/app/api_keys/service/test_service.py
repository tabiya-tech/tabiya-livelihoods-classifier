"""Service-layer tests with both Mongo and GCP fully mocked.

Asserts the orchestration contract:
- quota enforcement
- happy-path create returns plaintext only once
- create rolls back the GCP key if Mongo insert blows up
- revoke deletes from GCP before flipping the Mongo flag
"""

from dataclasses import dataclass, field

import pytest

from classify_v2.app.api_keys.repository.repository import IApiKeysRepository
from classify_v2.app.api_keys.service.errors import (
    ApiKeyNotFoundError,
    ApiKeysQuotaExceededError,
    GcpApiKeysError,
)
from classify_v2.app.api_keys.service.gcp_key_manager import IGcpKeyManager, IssuedGcpKey
from classify_v2.app.api_keys.service.service import ApiKeysService
from classify_v2.app.api_keys.service.types import ApiKeyMetadata


# ── Fakes ─────────────────────────────────────────────────────────────────


@dataclass
class FakeRepository(IApiKeysRepository):
    rows: list[dict] = field(default_factory=list)
    raise_on_insert: bool = False

    async def list_active(self, user_id: str) -> list[ApiKeyMetadata]:
        return [
            ApiKeyMetadata(**{k: v for k, v in row.items() if k in ApiKeyMetadata.model_fields})
            for row in self.rows
            if row["user_id"] == user_id and not row.get("revoked")
        ]

    async def count_active(self, user_id: str) -> int:
        return sum(1 for row in self.rows if row["user_id"] == user_id and not row.get("revoked"))

    async def insert(self, *, meta: ApiKeyMetadata, key_hash: str, gcp_key_name: str) -> None:
        if self.raise_on_insert:
            raise RuntimeError("simulated insert failure")
        row = meta.model_dump()
        row["key_hash"] = key_hash
        row["gcp_key_name"] = gcp_key_name
        self.rows.append(row)

    async def find_gcp_key_name(self, *, user_id: str, key_id: str) -> str | None:
        for row in self.rows:
            if row["user_id"] == user_id and row["key_id"] == key_id and not row.get("revoked"):
                return row.get("gcp_key_name")
        return None

    async def mark_revoked(self, *, user_id: str, key_id: str) -> bool:
        for row in self.rows:
            if row["user_id"] == user_id and row["key_id"] == key_id and not row.get("revoked"):
                row["revoked"] = True
                return True
        return False


@dataclass
class FakeGcpKeyManager(IGcpKeyManager):
    issued: list[IssuedGcpKey] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    raise_on_create: bool = False
    raise_on_delete: bool = False
    next_id: int = 1

    async def create_key(self, display_name: str) -> IssuedGcpKey:
        if self.raise_on_create:
            raise GcpApiKeysError("simulated GCP create failure")
        key_id = f"gcp-key-{self.next_id:03d}"
        self.next_id += 1
        issued = IssuedGcpKey(
            gcp_key_name=f"projects/test/locations/global/keys/{key_id}",
            key_id=key_id,
            key_string=f"plaintext-{key_id}",
        )
        self.issued.append(issued)
        return issued

    async def delete_key(self, gcp_key_name: str) -> None:
        if self.raise_on_delete:
            raise GcpApiKeysError("simulated GCP delete failure")
        self.deleted.append(gcp_key_name)


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_key_persists_metadata_and_returns_plaintext_once():
    # GIVEN a service with empty state
    repo = FakeRepository()
    gcp = FakeGcpKeyManager()
    svc = ApiKeysService(repository=repo, gcp=gcp, max_keys_per_user=5)

    # WHEN a user creates a key
    issued = await svc.create_key(user_id="uid-1", label="dev laptop")

    # THEN the plaintext is returned and the meta references the GCP key id
    assert issued.key == "plaintext-gcp-key-001"
    assert issued.meta.key_id == "gcp-key-001"
    assert issued.meta.label == "dev laptop"
    assert issued.meta.revoked is False
    # AND it shows up in the user's listing
    listed = await svc.list_keys(user_id="uid-1")
    assert len(listed) == 1
    assert listed[0].key_id == "gcp-key-001"


@pytest.mark.asyncio
async def test_create_key_rejects_when_quota_exceeded():
    # GIVEN a user already at the per-user quota
    repo = FakeRepository(
        rows=[
            {"user_id": "uid-1", "key_id": f"k{i}", "label": "x", "created_at": 0.0,
             "last_used_at": None, "revoked": False}
            for i in range(2)
        ]
    )
    gcp = FakeGcpKeyManager()
    svc = ApiKeysService(repository=repo, gcp=gcp, max_keys_per_user=2)

    # WHEN they try to create another
    # THEN the service refuses without ever calling GCP
    with pytest.raises(ApiKeysQuotaExceededError):
        await svc.create_key(user_id="uid-1", label="overflow")
    assert gcp.issued == []


@pytest.mark.asyncio
async def test_create_key_rolls_back_gcp_when_mongo_insert_fails():
    # GIVEN a repository that will reject the insert
    repo = FakeRepository(raise_on_insert=True)
    gcp = FakeGcpKeyManager()
    svc = ApiKeysService(repository=repo, gcp=gcp, max_keys_per_user=5)

    # WHEN create_key is called
    # THEN the underlying error propagates and the issued GCP key is deleted
    with pytest.raises(RuntimeError):
        await svc.create_key(user_id="uid-1", label="boom")
    assert len(gcp.issued) == 1
    assert gcp.deleted == [gcp.issued[0].gcp_key_name]


@pytest.mark.asyncio
async def test_revoke_deletes_from_gcp_then_marks_mongo():
    # GIVEN a user with one active key
    repo = FakeRepository()
    gcp = FakeGcpKeyManager()
    svc = ApiKeysService(repository=repo, gcp=gcp, max_keys_per_user=5)
    issued = await svc.create_key(user_id="uid-1", label="laptop")

    # WHEN they revoke it
    await svc.revoke_key(user_id="uid-1", key_id=issued.meta.key_id)

    # THEN the GCP key is deleted AND the row is no longer listed
    assert gcp.deleted == [f"projects/test/locations/global/keys/{issued.meta.key_id}"]
    assert await svc.list_keys(user_id="uid-1") == []


@pytest.mark.asyncio
async def test_revoke_raises_when_key_not_found():
    # GIVEN no keys for this user
    svc = ApiKeysService(
        repository=FakeRepository(),
        gcp=FakeGcpKeyManager(),
        max_keys_per_user=5,
    )

    # WHEN revoking an unknown key id
    # THEN ApiKeyNotFoundError surfaces and GCP is never called
    with pytest.raises(ApiKeyNotFoundError):
        await svc.revoke_key(user_id="uid-1", key_id="ghost")


@pytest.mark.asyncio
async def test_revoke_does_not_mark_mongo_when_gcp_delete_fails():
    # GIVEN a user with a key and a GCP client that rejects deletes
    repo = FakeRepository()
    gcp = FakeGcpKeyManager()
    svc = ApiKeysService(repository=repo, gcp=gcp, max_keys_per_user=5)
    issued = await svc.create_key(user_id="uid-1", label="laptop")
    gcp.raise_on_delete = True

    # WHEN revoke is called
    # THEN the GCP error propagates and the row remains active for retry
    with pytest.raises(GcpApiKeysError):
        await svc.revoke_key(user_id="uid-1", key_id=issued.meta.key_id)
    assert len(await svc.list_keys(user_id="uid-1")) == 1

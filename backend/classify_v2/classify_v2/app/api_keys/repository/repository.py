"""api_keys MongoDB repository.

The collection layout matches v1 so we can run the two services side by side
during migration. Soft-delete via `revoked: true` mirrors v1 — listing
filters out revoked rows; deletion never removes the doc.
"""

from abc import ABC, abstractmethod

from motor.motor_asyncio import AsyncIOMotorDatabase

from classify_v2.app.api_keys.service.types import ApiKeyMetadata

_COLLECTION = "api_keys"


class IApiKeysRepository(ABC):
    @abstractmethod
    async def list_active(self, user_id: str) -> list[ApiKeyMetadata]: ...

    @abstractmethod
    async def count_active(self, user_id: str) -> int: ...

    @abstractmethod
    async def insert(
        self,
        *,
        meta: ApiKeyMetadata,
        key_hash: str,
        gcp_key_name: str,
    ) -> None: ...

    @abstractmethod
    async def find_gcp_key_name(self, *, user_id: str, key_id: str) -> str | None:
        """Return the GCP resource name for the user's key, or None if absent."""

    @abstractmethod
    async def mark_revoked(self, *, user_id: str, key_id: str) -> bool: ...


class ApiKeysRepository(IApiKeysRepository):
    def __init__(self, app_db: AsyncIOMotorDatabase) -> None:
        self._col = app_db[_COLLECTION]

    async def list_active(self, user_id: str) -> list[ApiKeyMetadata]:
        cursor = self._col.find(
            {"user_id": user_id, "revoked": {"$ne": True}},
            {"_id": 0, "key_hash": 0, "gcp_key_name": 0},
        )
        docs = await cursor.to_list(length=None)
        return [ApiKeyMetadata(**doc) for doc in docs]

    async def count_active(self, user_id: str) -> int:
        return await self._col.count_documents(
            {"user_id": user_id, "revoked": {"$ne": True}}
        )

    async def insert(
        self,
        *,
        meta: ApiKeyMetadata,
        key_hash: str,
        gcp_key_name: str,
    ) -> None:
        doc = meta.model_dump()
        doc["key_hash"] = key_hash
        doc["gcp_key_name"] = gcp_key_name
        await self._col.insert_one(doc)

    async def find_gcp_key_name(self, *, user_id: str, key_id: str) -> str | None:
        doc = await self._col.find_one(
            {"user_id": user_id, "key_id": key_id, "revoked": {"$ne": True}},
            {"gcp_key_name": 1, "_id": 0},
        )
        if not doc:
            return None
        return doc.get("gcp_key_name")

    async def mark_revoked(self, *, user_id: str, key_id: str) -> bool:
        result = await self._col.update_one(
            {"user_id": user_id, "key_id": key_id, "revoked": {"$ne": True}},
            {"$set": {"revoked": True}},
        )
        return result.modified_count > 0

"""User config repository — reads and writes user_configs collection."""

from abc import ABC, abstractmethod

from motor.motor_asyncio import AsyncIOMotorDatabase

from nel.app.user_config.service.types import UserConfig

_COLLECTION = "user_configs"


class IUserConfigRepository(ABC):
    @abstractmethod
    async def get(self, user_id: str) -> UserConfig | None: ...

    @abstractmethod
    async def upsert(self, config: UserConfig) -> None: ...


class UserConfigRepository(IUserConfigRepository):
    def __init__(self, app_db: AsyncIOMotorDatabase):
        self._col = app_db[_COLLECTION]

    async def get(self, user_id: str) -> UserConfig | None:
        doc = await self._col.find_one({"user_id": user_id})
        if doc is None:
            return None
        return UserConfig(
            user_id=user_id,
            taxonomy_model_id=doc.get("taxonomy_model_id", ""),
            nel_model_id=doc.get("nel_model_id") or doc.get("nel_model_name", ""),
        )

    async def upsert(self, config: UserConfig) -> None:
        await self._col.update_one(
            {"user_id": config.user_id},
            {"$set": {
                "taxonomy_model_id": config.taxonomy_model_id,
                "nel_model_id": config.nel_model_id,
            }},
            upsert=True,
        )

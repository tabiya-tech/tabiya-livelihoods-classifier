"""NEL model info repository."""

import logging
from abc import ABC, abstractmethod

from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel

from nel.app.server_dependencies.database_collections import Collections

_logger = logging.getLogger(__name__)


class NELModelInfo(BaseModel):
    model_id: str
    dimensions: int
    description: str = ""


class INELModelRepository(ABC):
    @abstractmethod
    async def get_all(self) -> list[NELModelInfo]: ...

    @abstractmethod
    async def get(self, model_id: str) -> NELModelInfo | None: ...

    @abstractmethod
    async def upsert(self, model: NELModelInfo) -> None: ...


class NELModelRepository(INELModelRepository):
    def __init__(self, app_db: AsyncIOMotorDatabase):
        self._col = app_db[Collections.NEL_MODELS]

    async def get_all(self) -> list[NELModelInfo]:
        return [
            NELModelInfo(**{k: v for k, v in doc.items() if k != "_id"})
            async for doc in self._col.find({})
        ]

    async def get(self, model_id: str) -> NELModelInfo | None:
        doc = await self._col.find_one({"model_id": {"$eq": model_id}})
        if doc is None:
            return None
        doc.pop("_id", None)
        return NELModelInfo(**doc)

    async def upsert(self, model: NELModelInfo) -> None:
        await self._col.update_one(
            {"model_id": {"$eq": model.model_id}},
            {"$set": model.model_dump()},
            upsert=True,
        )

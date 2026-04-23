"""Embeddings cache repository.

Manages two concerns:
1. Cache status tracking (app DB) — which (taxonomy_model_id, nel_model_id, entity_type)
   combinations have been generated and are ready for vector search.
2. Embedding storage and retrieval (taxonomy DB) — the actual embedding documents,
   queried via MongoDB Atlas $vectorSearch aggregation.
"""

import logging
from abc import ABC, abstractmethod

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import BulkWriteError

from nel.app.embeddings_cache.types import EmbeddingCacheStatus, EmbeddingDocument, CacheStatus
from nel.app.server_dependencies.database_collections import Collections

_logger = logging.getLogger(__name__)

_ENTITY_TYPE_TO_COLLECTION = {
    "occupation": Collections.OCCUPATION_EMBEDDINGS,
    "skill": Collections.SKILL_EMBEDDINGS,
    "qualification": Collections.QUALIFICATION_EMBEDDINGS,
}


class IEmbeddingsCacheRepository(ABC):
    @abstractmethod
    async def get_cache_status(
        self, taxonomy_model_id: str, nel_model_id: str, entity_type: str
    ) -> EmbeddingCacheStatus | None: ...

    @abstractmethod
    async def upsert_cache_status(self, status: EmbeddingCacheStatus) -> None: ...

    @abstractmethod
    async def delete_embeddings(
        self, taxonomy_model_id: str, nel_model_id: str, entity_type: str
    ) -> int: ...

    @abstractmethod
    async def insert_embeddings_batch(
        self, entity_type: str, documents: list[EmbeddingDocument]
    ) -> int: ...

    @abstractmethod
    async def vector_search(
        self,
        entity_type: str,
        query_embedding: list[float],
        taxonomy_model_id: str,
        nel_model_id: str,
        top_k: int,
        min_similarity: float,
        index_name: str,
    ) -> list[tuple[EmbeddingDocument, float]]: ...

    @abstractmethod
    async def ensure_indexes(self) -> None: ...


class EmbeddingsCacheRepository(IEmbeddingsCacheRepository):
    def __init__(self, app_db: AsyncIOMotorDatabase, taxonomy_db: AsyncIOMotorDatabase):
        self._app_db = app_db
        self._taxonomy_db = taxonomy_db

    def _status_col(self):
        return self._app_db[Collections.EMBEDDINGS_CACHE_STATUS]

    def _embedding_col(self, entity_type: str):
        col_name = _ENTITY_TYPE_TO_COLLECTION[entity_type]
        return self._taxonomy_db[col_name]

    async def get_cache_status(
        self, taxonomy_model_id: str, nel_model_id: str, entity_type: str
    ) -> EmbeddingCacheStatus | None:
        doc = await self._status_col().find_one({
            "taxonomy_model_id": {"$eq": taxonomy_model_id},
            "nel_model_id": {"$eq": nel_model_id},
            "entity_type": {"$eq": entity_type},
        })
        if doc is None:
            return None
        doc.pop("_id", None)
        return EmbeddingCacheStatus(**doc)

    async def upsert_cache_status(self, status: EmbeddingCacheStatus) -> None:
        await self._status_col().update_one(
            {
                "taxonomy_model_id": {"$eq": status.taxonomy_model_id},
                "nel_model_id": {"$eq": status.nel_model_id},
                "entity_type": {"$eq": status.entity_type},
            },
            {"$set": status.model_dump()},
            upsert=True,
        )

    async def delete_embeddings(
        self, taxonomy_model_id: str, nel_model_id: str, entity_type: str
    ) -> int:
        result = await self._embedding_col(entity_type).delete_many({
            "taxonomy_model_id": {"$eq": taxonomy_model_id},
            "nel_model_id": {"$eq": nel_model_id},
        })
        return result.deleted_count

    async def insert_embeddings_batch(
        self, entity_type: str, documents: list[EmbeddingDocument]
    ) -> int:
        """Insert documents, returning the count of successfully inserted docs.

        Uses ordered=False so MongoDB continues past individual write errors
        (e.g. duplicate keys on re-runs). Per-document errors are logged but
        do not abort the batch. Raises on infrastructure-level failures
        (network, auth) so the caller's retry policy can handle them.
        """
        if not documents:
            return 0
        docs = [doc.model_dump() for doc in documents]
        try:
            result = await self._embedding_col(entity_type).insert_many(docs, ordered=False)
            return len(result.inserted_ids)
        except BulkWriteError as exc:
            # BulkWriteError is raised even when some docs succeeded (ordered=False).
            # nInserted tells us how many actually made it.
            inserted = exc.details.get("nInserted", 0)
            errors = exc.details.get("writeErrors", [])
            # Duplicate key (code 11000) is expected on re-runs — log at debug level.
            # Anything else is worth a warning.
            for err in errors:
                if err.get("code") == 11000:
                    _logger.debug(
                        "Skipped duplicate %s uuid=%s",
                        entity_type,
                        err.get("keyValue", {}).get("entity_uuid", "?"),
                    )
                else:
                    _logger.warning(
                        "Insert error for %s doc index %d (code %d): %s",
                        entity_type, err.get("index", -1), err.get("code", -1), err.get("errmsg", ""),
                    )
            return inserted

    async def vector_search(
        self,
        entity_type: str,
        query_embedding: list[float],
        taxonomy_model_id: str,
        nel_model_id: str,
        top_k: int,
        min_similarity: float,
        index_name: str = "nel_embedding_index_384",
    ) -> list[tuple[EmbeddingDocument, float]]:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                    "filter": {
                        "taxonomy_model_id": {"$eq": taxonomy_model_id},
                        "nel_model_id": {"$eq": nel_model_id},
                    },
                }
            },
            {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
        ]
        results = []
        async for doc in self._embedding_col(entity_type).aggregate(pipeline):
            score = doc.pop("_score", 0.0)
            doc.pop("_id", None)
            if score >= min_similarity:
                results.append((EmbeddingDocument(**doc), score))
        return results

    async def ensure_indexes(self) -> None:
        """Create standard (non-vector) MongoDB indexes. Safe to call repeatedly."""
        # Cache status: unique per (taxonomy_model_id, nel_model_id, entity_type)
        await self._status_col().create_index(
            [("taxonomy_model_id", 1), ("nel_model_id", 1), ("entity_type", 1)],
            unique=True,
        )
        # Embedding collections: unique per entity_uuid within a (model, taxonomy) combo
        for entity_type in _ENTITY_TYPE_TO_COLLECTION:
            col = self._embedding_col(entity_type)
            await col.create_index(
                [("taxonomy_model_id", 1), ("nel_model_id", 1), ("entity_uuid", 1)],
                unique=True,
            )
        _logger.info("Embeddings cache indexes ensured")

"""MongoDB provider for the nel-v2 service.

Two databases:
- Application DB: nel_models, nel_qualifications, nel_embeddings_cache_status
- Taxonomy DB:    nel_occupation_embeddings, nel_skill_embeddings,
                  nel_qualification_embeddings  (Atlas vector search)

Uses the double-checked locking pattern (same as Compass / classify service) to
lazily initialise each AsyncIOMotorClient exactly once.
"""

import asyncio
import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

_logger = logging.getLogger(__name__)


# ── Factory functions (decoupled so tests can patch them) ─────────────────

def _create_application_db(mongodb_uri: str, db_name: str) -> AsyncIOMotorDatabase:
    return AsyncIOMotorClient(mongodb_uri, tlsAllowInvalidCertificates=True).get_database(db_name)


def _create_taxonomy_db(mongodb_uri: str, db_name: str) -> AsyncIOMotorDatabase:
    return AsyncIOMotorClient(mongodb_uri, tlsAllowInvalidCertificates=True).get_database(db_name)


# ── Provider ──────────────────────────────────────────────────────────────

class ClassifierDBProvider:
    _application_db: AsyncIOMotorDatabase | None = None
    _taxonomy_db: AsyncIOMotorDatabase | None = None
    _lock = asyncio.Lock()

    # ── Application DB ────────────────────────────────────────────────────

    @staticmethod
    def _application_settings() -> tuple[str, str]:
        from nel.config import APPLICATION_MONGODB_URI, APPLICATION_DATABASE_NAME
        return APPLICATION_MONGODB_URI, APPLICATION_DATABASE_NAME

    @classmethod
    async def get_application_db(cls) -> AsyncIOMotorDatabase:
        if cls._application_db is None:
            async with cls._lock:
                if cls._application_db is None:
                    uri, name = cls._application_settings()
                    _logger.info("Connecting to application MongoDB (db: %s)", name)
                    cls._application_db = _create_application_db(uri, name)
                    ok = await _ping(cls._application_db.client)
                    if not ok:
                        cls._application_db = None
                        raise RuntimeError("Application MongoDB health check failed")
                    _logger.info("Connected to application MongoDB")
        return cls._application_db

    # ── Taxonomy DB ───────────────────────────────────────────────────────

    @staticmethod
    def _taxonomy_settings() -> tuple[str, str]:
        from nel.config import TAXONOMY_MONGODB_URI, TAXONOMY_DATABASE_NAME
        return TAXONOMY_MONGODB_URI, TAXONOMY_DATABASE_NAME

    @classmethod
    async def get_taxonomy_db(cls) -> AsyncIOMotorDatabase:
        if cls._taxonomy_db is None:
            async with cls._lock:
                if cls._taxonomy_db is None:
                    uri, name = cls._taxonomy_settings()
                    _logger.info("Connecting to taxonomy MongoDB (db: %s)", name)
                    cls._taxonomy_db = _create_taxonomy_db(uri, name)
                    ok = await _ping(cls._taxonomy_db.client)
                    if not ok:
                        cls._taxonomy_db = None
                        raise RuntimeError("Taxonomy MongoDB health check failed")
                    _logger.info("Connected to taxonomy MongoDB")
        return cls._taxonomy_db

    # ── Utilities ─────────────────────────────────────────────────────────

    @classmethod
    def clear_cache(cls) -> None:
        """Reset cached instances. Used in tests."""
        cls._application_db = None
        cls._taxonomy_db = None


async def _ping(client: AsyncIOMotorClient) -> bool:
    try:
        result = await client.admin.command("ping")
        return result.get("ok") == 1.0
    except Exception:
        return False

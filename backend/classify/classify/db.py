"""MongoDB provider for the classify service.

Uses the double-checked locking pattern (same as Compass) to lazily
initialise the AsyncIOMotorClient exactly once across all concurrent requests.

Usage:
    db = await ApplicationDBProvider.get_db()
    collection = db["my_collection"]
"""

import asyncio
import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

_logger = logging.getLogger(__name__)


def _create_db(mongodb_uri: str, db_name: str) -> AsyncIOMotorDatabase:
    """Create the motor database instance.

    Decoupled from the provider so tests can patch this function
    instead of the provider itself.
    """
    return AsyncIOMotorClient(mongodb_uri, tlsAllowInvalidCertificates=True).get_database(db_name)


class ApplicationDBProvider:
    _db: AsyncIOMotorDatabase | None = None
    _lock = asyncio.Lock()

    @staticmethod
    def _settings() -> tuple[str, str]:
        # Deferred import so env vars are not required at import time.
        from classify.config import MONGODB_URI, MONGODB_DB_NAME
        return MONGODB_URI, MONGODB_DB_NAME

    @classmethod
    async def get_db(cls) -> AsyncIOMotorDatabase:
        if cls._db is None:
            async with cls._lock:
                if cls._db is None:
                    mongodb_uri, db_name = cls._settings()
                    _logger.info("Connecting to MongoDB (db: %s)", db_name)
                    cls._db = _create_db(mongodb_uri, db_name)
                    ok = await _ping(cls._db.client)
                    if not ok:
                        cls._db = None
                        raise RuntimeError("MongoDB health check failed")
                    _logger.info("Connected to MongoDB")
        return cls._db

    @classmethod
    def clear_cache(cls) -> None:
        """Reset the cached instance. Used in tests."""
        cls._db = None


async def _ping(client: AsyncIOMotorClient) -> bool:
    try:
        result = await client.admin.command("ping")
        return result.get("ok") == 1.0
    except Exception:
        return False

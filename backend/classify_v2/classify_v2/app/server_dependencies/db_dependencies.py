"""Application MongoDB provider for the classify-v2 service.

Lazily initialises a single AsyncIOMotorClient for the application DB
(api_keys collection lives here; future user_configs etc. will too).

Mirrors nel-v2's provider so the two services share an operational shape.
"""

import asyncio
import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

_logger = logging.getLogger(__name__)


def _create_application_db(mongodb_uri: str, db_name: str) -> AsyncIOMotorDatabase:
    from classify_v2.config import TARGET_ENVIRONMENT_TYPE
    tls_allow_invalid = TARGET_ENVIRONMENT_TYPE == "local"
    return AsyncIOMotorClient(mongodb_uri, tlsAllowInvalidCertificates=tls_allow_invalid).get_database(db_name)


class ClassifyDBProvider:
    _application_db: AsyncIOMotorDatabase | None = None
    _lock = asyncio.Lock()

    @staticmethod
    def _application_settings() -> tuple[str, str]:
        from classify_v2.config import APPLICATION_MONGODB_URI, APPLICATION_DATABASE_NAME
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

    @classmethod
    def clear_cache(cls) -> None:
        """Reset cached instances. Used in tests."""
        cls._application_db = None


async def _ping(client: AsyncIOMotorClient) -> bool:
    try:
        result = await client.admin.command("ping")
        return result.get("ok") == 1.0
    except Exception:
        return False

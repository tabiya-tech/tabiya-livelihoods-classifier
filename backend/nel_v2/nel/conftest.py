import logging
import platform
import random
import string
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from nel.app.server_dependencies.db_dependencies import ClassifierDBProvider


# ── In-memory MongoDB server (session-scoped — start once per test run) ────

@pytest.fixture(scope="session")
def in_memory_mongo_server():
    from pymongo_inmemory import Mongod
    from pymongo_inmemory.context import Context

    # Workaround: pymongo_inmemory falls back to Mongo 4.0.23 on Ubuntu/Debian.
    # https://github.com/kaizendorks/pymongo_inmemory/issues/115
    os_name: str | None = None
    version_str = platform.uname().version.lower()
    if "ubuntu" in version_str:
        os_name = "ubuntu"
    elif "debian" in version_str:
        os_name = "debian"

    ctx = Context(version="7.0", os_name=os_name)
    # Workaround: pymongo_inmemory incorrectly sets storage engine to
    # "ephemeralForTest" for Mongo >= 6. https://github.com/kaizendorks/pymongo_inmemory/pull/119
    ctx.storage_engine = "wiredTiger"

    server = Mongod(ctx)
    server.start()
    yield server
    logging.info("Stopping in-memory MongoDB server")
    server.stop()


def _random_db_name() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=10))  # nosec B311


@pytest.fixture(scope="function")
def in_memory_application_db_name(in_memory_mongo_server) -> str:
    """Unique application DB name per test, pointing at the in-memory server."""
    return _random_db_name()


@pytest.fixture(scope="function")
def in_memory_taxonomy_db_name(in_memory_mongo_server) -> str:
    """Unique taxonomy DB name per test, pointing at the in-memory server."""
    return _random_db_name()


# ── Motor database fixtures ────────────────────────────────────────────────

@pytest.fixture(scope="function")
async def in_memory_application_database(
    in_memory_mongo_server,
    in_memory_application_db_name,
) -> AsyncIOMotorDatabase:
    """A fresh Motor database per test, backed by the in-memory server."""
    return AsyncIOMotorClient(
        in_memory_mongo_server.connection_string,
        tlsAllowInvalidCertificates=True,
    ).get_database(in_memory_application_db_name)


@pytest.fixture(scope="function")
async def in_memory_taxonomy_database(
    in_memory_mongo_server,
    in_memory_taxonomy_db_name,
) -> AsyncIOMotorDatabase:
    """A fresh Motor taxonomy database per test, backed by the in-memory server."""
    return AsyncIOMotorClient(
        in_memory_mongo_server.connection_string,
        tlsAllowInvalidCertificates=True,
    ).get_database(in_memory_taxonomy_db_name)


# ── Context manager for patching both DB factories ────────────────────────

@contextmanager
def patch_db_provider(in_memory_mongo_server, application_db_name: str, taxonomy_db_name: str):
    """
    Patch both _create_application_db and _create_taxonomy_db to route to the
    in-memory MongoDB server. Resets ClassifierDBProvider cache before and after.
    """
    connection_string = in_memory_mongo_server.connection_string

    def _make_application_db(_uri, _name):
        return AsyncIOMotorClient(
            connection_string,
            tlsAllowInvalidCertificates=True,
        ).get_database(application_db_name)

    def _make_taxonomy_db(_uri, _name):
        return AsyncIOMotorClient(
            connection_string,
            tlsAllowInvalidCertificates=True,
        ).get_database(taxonomy_db_name)

    ClassifierDBProvider.clear_cache()
    with patch("nel.app.server_dependencies.db_dependencies._create_application_db", side_effect=_make_application_db):
        with patch("nel.app.server_dependencies.db_dependencies._create_taxonomy_db", side_effect=_make_taxonomy_db):
            yield
    ClassifierDBProvider.clear_cache()

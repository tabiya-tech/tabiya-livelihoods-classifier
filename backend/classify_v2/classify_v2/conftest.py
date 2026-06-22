"""Shared test fixtures for classify_v2."""

import logging
import platform
import random
import string

import pytest
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase


@pytest.fixture(scope="session")
def in_memory_mongo_server():
    from pymongo_inmemory import Mongod
    from pymongo_inmemory.context import Context

    # See nel_v2/conftest.py for the same workarounds.
    os_name: str | None = None
    version_str = platform.uname().version.lower()
    if "ubuntu" in version_str:
        os_name = "ubuntu"
    elif "debian" in version_str:
        os_name = "debian"

    ctx = Context(version="7.0", os_name=os_name)
    ctx.storage_engine = "wiredTiger"

    server = Mongod(ctx)
    server.start()
    yield server
    logging.info("Stopping in-memory MongoDB server")
    server.stop()


def _random_db_name() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=10))  # nosec B311


@pytest.fixture(scope="function")
async def in_memory_application_database(in_memory_mongo_server) -> AsyncIOMotorDatabase:
    """A fresh Motor database per test, backed by the in-memory server."""
    return AsyncIOMotorClient(
        in_memory_mongo_server.connection_string,
        tlsAllowInvalidCertificates=True,
    ).get_database(_random_db_name())

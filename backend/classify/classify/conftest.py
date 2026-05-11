import logging
import platform
import random
import string
from typing import Generator
from unittest.mock import patch

import pytest
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from classify.db import ApplicationDBProvider
from classify.service import IClassifyService
from classify.get_classify_service import get_classify_service
from classify.user_config import get_api_key_user
from classify.main import app


def _silence_pymongo_inmemory_loggers() -> None:
    """Avoid pymongo_inmemory atexit logging to a stream pytest has already closed (I/O on closed file).

    pymongo_inmemory/mongod.py uses logger name ``PYMONGOIM_MONGOD`` (not ``pymongo_inmemory.*``).
    """
    for name in (
        "PYMONGOIM_MONGOD",
        "pymongo_inmemory",
        "pymongo_inmemory.mongod",
        "pymongo_inmemory.context",
    ):
        log = logging.getLogger(name)
        log.handlers.clear()
        log.addHandler(logging.NullHandler())
        log.propagate = False


_silence_pymongo_inmemory_loggers()

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
def in_memory_db_name(in_memory_mongo_server) -> str:
    """Unique database name per test, pointing at the in-memory server."""
    return _random_db_name()


# ── FastAPI test client helpers ────────────────────────────────────────────

class MockClassifyService(IClassifyService):
    async def classify(self, input_text, options=None):
        raise NotImplementedError()


def _mock_api_key_user():
    return {"user_id": "test-user"}


@pytest.fixture(scope="function")
def client_with_mocks(in_memory_mongo_server, in_memory_db_name) -> Generator[tuple, None, None]:
    """TestClient wired to in-memory MongoDB and a mock classify service."""
    mock_service = MockClassifyService()
    app.dependency_overrides[get_classify_service] = lambda: mock_service
    app.dependency_overrides[get_api_key_user] = _mock_api_key_user

    # Patch _create_db to return a fresh motor client bound to the in-memory
    # server. The client is created lazily inside the TestClient's event loop
    # so motor binds to the right loop.
    connection_string = in_memory_mongo_server.connection_string
    db_name = in_memory_db_name

    def _make_in_memory_db(_uri, _name):
        return AsyncIOMotorClient(
            connection_string,
            tlsAllowInvalidCertificates=True,
        ).get_database(db_name)

    ApplicationDBProvider.clear_cache()
    with patch("classify.db._create_db", side_effect=_make_in_memory_db):
        from fastapi.testclient import TestClient
        with TestClient(app) as client:
            yield client, mock_service

    app.dependency_overrides.clear()
    ApplicationDBProvider.clear_cache()

import pytest
from fastapi.testclient import TestClient
from typing import Generator

from nel.service import INELService
from nel.get_nel_service import get_nel_service
from nel.main import app


class MockNELService(INELService):
    def link_entities(self, entities, options=None):
        raise NotImplementedError()


def _create_test_client_with_mocks() -> tuple[TestClient, INELService]:
    mock_service = MockNELService()
    app.dependency_overrides[get_nel_service] = lambda: mock_service
    return TestClient(app), mock_service


@pytest.fixture(scope="function")
def client_with_mocks() -> Generator[tuple[TestClient, INELService], None, None]:
    client, mock_service = _create_test_client_with_mocks()
    yield client, mock_service
    app.dependency_overrides.clear()

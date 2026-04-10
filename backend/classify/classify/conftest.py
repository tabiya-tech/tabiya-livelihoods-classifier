import pytest
from fastapi.testclient import TestClient
from typing import Generator

from classify.service import IClassifyService
from classify.get_classify_service import get_classify_service
from classify.user_config import get_api_key_user
from classify.main import app


class MockClassifyService(IClassifyService):
    async def classify(self, input_text, options=None):
        raise NotImplementedError()


def _mock_api_key_user():
    return {"user_id": "test-user"}


def _create_test_client_with_mocks() -> tuple[TestClient, IClassifyService]:
    mock_service = MockClassifyService()
    app.dependency_overrides[get_classify_service] = lambda: mock_service
    app.dependency_overrides[get_api_key_user] = _mock_api_key_user
    return TestClient(app), mock_service


@pytest.fixture(scope="function")
def client_with_mocks() -> Generator[tuple[TestClient, IClassifyService], None, None]:
    client, mock_service = _create_test_client_with_mocks()
    yield client, mock_service
    app.dependency_overrides.clear()

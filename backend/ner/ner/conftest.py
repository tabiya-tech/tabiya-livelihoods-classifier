import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typing import Generator

from ner.service import INERService
from ner.get_ner_service import get_ner_service
from ner.main import app


class MockNERService(INERService):
    def extract_entities(self, text, entity_types=None):
        raise NotImplementedError()


def _create_test_client_with_mocks() -> tuple[TestClient, INERService]:
    mock_service = MockNERService()
    app.dependency_overrides[get_ner_service] = lambda: mock_service
    return TestClient(app), mock_service


@pytest.fixture(scope="function")
def client_with_mocks() -> Generator[tuple[TestClient, INERService], None, None]:
    client, mock_service = _create_test_client_with_mocks()
    yield client, mock_service
    app.dependency_overrides.clear()

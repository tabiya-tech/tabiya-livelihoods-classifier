"""Tests for NELModelRepository."""

import pytest

from nel.app.nel_models.repository.repository import NELModelInfo, NELModelRepository
from nel.conftest import patch_db_provider


@pytest.fixture
async def repo(in_memory_application_database):
    return NELModelRepository(app_db=in_memory_application_database)


class TestNELModelRepository:
    async def test_get_returns_none_when_not_exists(self, repo):
        result = await repo.get("nonexistent")
        assert result is None

    async def test_upsert_then_get(self, repo):
        # GIVEN a model is upserted
        model = NELModelInfo(model_id="all-MiniLM-L6-v2", dimensions=384, description="Test")
        await repo.upsert(model)

        # WHEN retrieved
        result = await repo.get("all-MiniLM-L6-v2")

        # THEN it matches
        assert result is not None
        assert result.model_id == "all-MiniLM-L6-v2"
        assert result.dimensions == 384

    async def test_upsert_updates_existing(self, repo):
        # GIVEN a model with dimensions 384
        await repo.upsert(NELModelInfo(model_id="m1", dimensions=384))

        # WHEN upserted again with different dimensions
        await repo.upsert(NELModelInfo(model_id="m1", dimensions=768))

        # THEN the updated value is returned
        result = await repo.get("m1")
        assert result.dimensions == 768

    async def test_get_all_returns_all(self, repo):
        # GIVEN two models
        await repo.upsert(NELModelInfo(model_id="m1", dimensions=384))
        await repo.upsert(NELModelInfo(model_id="m2", dimensions=768))

        # WHEN get_all
        results = await repo.get_all()

        # THEN both returned
        assert len(results) == 2
        assert {r.model_id for r in results} == {"m1", "m2"}

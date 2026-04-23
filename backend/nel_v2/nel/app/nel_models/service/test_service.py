"""Tests for NELModelService."""

import pytest

from nel.app.nel_models.repository.repository import INELModelRepository, NELModelInfo
from nel.app.nel_models.service.errors import NELModelNotFoundError
from nel.app.nel_models.service.service import NELModelService


class FakeRepo(INELModelRepository):
    def __init__(self, models: list[NELModelInfo] | None = None):
        self._models = {m.model_id: m for m in (models or [])}

    async def get_all(self):
        return list(self._models.values())

    async def get(self, model_id):
        return self._models.get(model_id)

    async def upsert(self, model):
        self._models[model.model_id] = model


class TestNELModelServiceGetAll:
    async def test_returns_all_models(self):
        # GIVEN two models exist
        repo = FakeRepo([
            NELModelInfo(model_id="m1", dimensions=384),
            NELModelInfo(model_id="m2", dimensions=768),
        ])
        svc = NELModelService(repo)

        # WHEN get_all is called
        result = await svc.get_all()

        # THEN both models are returned
        assert len(result) == 2
        assert {m.model_id for m in result} == {"m1", "m2"}

    async def test_returns_empty_list_when_no_models(self):
        # GIVEN no models exist
        svc = NELModelService(FakeRepo())

        # WHEN get_all is called
        result = await svc.get_all()

        # THEN an empty list is returned
        assert result == []


class TestNELModelServiceGet:
    async def test_returns_model_when_exists(self):
        # GIVEN a model exists
        repo = FakeRepo([NELModelInfo(model_id="m1", dimensions=384)])
        svc = NELModelService(repo)

        # WHEN get is called with its id
        result = await svc.get("m1")

        # THEN the correct model is returned
        assert result.model_id == "m1"
        assert result.dimensions == 384

    async def test_raises_when_model_not_found(self):
        # GIVEN no models exist
        svc = NELModelService(FakeRepo())

        # WHEN get is called with an unknown id
        with pytest.raises(NELModelNotFoundError) as exc_info:
            await svc.get("nonexistent")

        # THEN the error contains the model id
        assert "nonexistent" in str(exc_info.value)

    async def test_raises_for_wrong_id_when_others_exist(self):
        # GIVEN one model exists
        repo = FakeRepo([NELModelInfo(model_id="m1", dimensions=384)])
        svc = NELModelService(repo)

        # WHEN get is called with a different id
        with pytest.raises(NELModelNotFoundError):
            await svc.get("m2")

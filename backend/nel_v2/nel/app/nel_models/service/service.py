"""NEL model service."""

import logging
from abc import ABC, abstractmethod

from nel.app.nel_models.repository.repository import INELModelRepository, NELModelInfo
from nel.app.nel_models.service.errors import NELModelNotFoundError

_logger = logging.getLogger(__name__)


class INELModelService(ABC):
    @abstractmethod
    async def get_all(self) -> list[NELModelInfo]: ...

    @abstractmethod
    async def get(self, model_id: str) -> NELModelInfo: ...


class NELModelService(INELModelService):
    def __init__(self, repository: INELModelRepository):
        self._repo = repository

    async def get_all(self) -> list[NELModelInfo]:
        return await self._repo.get_all()

    async def get(self, model_id: str) -> NELModelInfo:
        model = await self._repo.get(model_id)
        if model is None:
            raise NELModelNotFoundError(model_id)
        return model

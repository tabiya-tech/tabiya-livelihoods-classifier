"""User config service."""

import logging
from abc import ABC, abstractmethod

from nel.app.user_config.repository.repository import IUserConfigRepository
from nel.app.user_config.service.types import UserConfig

_logger = logging.getLogger(__name__)


class IUserConfigService(ABC):
    @abstractmethod
    async def get(self, user_id: str) -> UserConfig: ...

    @abstractmethod
    async def upsert(self, config: UserConfig) -> UserConfig: ...


class UserConfigService(IUserConfigService):
    def __init__(self, repository: IUserConfigRepository):
        self._repo = repository

    async def get(self, user_id: str) -> UserConfig:
        config = await self._repo.get(user_id)
        if config is None:
            # Return an empty config — user hasn't set preferences yet
            return UserConfig(user_id=user_id, taxonomy_model_id="", nel_model_id="")
        return config

    async def upsert(self, config: UserConfig) -> UserConfig:
        await self._repo.upsert(config)
        return config

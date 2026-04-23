"""Taxonomy models service — proxies the taxonomy REST API."""

import logging
from abc import ABC, abstractmethod

import httpx

from nel.app.taxonomy_models.service.errors import TaxonomyAPIError
from nel.app.taxonomy_models.service.types import TaxonomyModelInfo

_logger = logging.getLogger(__name__)


class ITaxonomyModelService(ABC):
    @abstractmethod
    async def get_all(self) -> list[TaxonomyModelInfo]: ...


class TaxonomyModelService(ITaxonomyModelService):
    def __init__(self, base_url: str, api_key: str = ""):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def _headers(self) -> dict:
        if self._api_key:
            return {"X-API-Key": self._api_key}
        return {}

    async def get_all(self) -> list[TaxonomyModelInfo]:
        url = f"{self._base_url}/api/app/models"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=self._headers())
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise TaxonomyAPIError(
                f"Taxonomy API returned {exc.response.status_code} for {url}"
            ) from exc
        except httpx.TransportError as exc:
            raise TaxonomyAPIError(f"Taxonomy API unreachable: {exc}") from exc

        models = []
        for item in resp.json():
            models.append(TaxonomyModelInfo(
                id=item["id"],
                name=item.get("name", ""),
                version=item.get("version", ""),
                description=item.get("description", ""),
                released=item.get("released", False),
            ))
        return models

"""Taxonomy data sources.

TaxonomyAPISource — fetches occupations and skills from the taxonomy REST API
                    (paginated, async).
QualificationsMongoSource — reads qualifications from the nel_qualifications
                            MongoDB collection (seeded once from CSV).
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator

import httpx
from motor.motor_asyncio import AsyncIOMotorDatabase

from nel.app.server_dependencies.database_collections import Collections

_logger = logging.getLogger(__name__)

_PAGE_LIMIT = 100  # API maximum per page


class ITaxonomySource(ABC):
    @abstractmethod
    def fetch_occupations(self, taxonomy_model_id: str) -> AsyncIterator[dict]: ...

    @abstractmethod
    def fetch_skills(self, taxonomy_model_id: str) -> AsyncIterator[dict]: ...


class TaxonomyAPISource(ITaxonomySource):
    """Paginates GET /models/{id}/occupations|skills using cursor-based pagination.

    Auth: X-API-Key header.
    Pagination: each response has {"data": [...], "nextCursor": "<base64>|null"}.
    """

    def __init__(self, base_url: str, api_key: str = ""):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def _headers(self) -> dict:
        if self._api_key:
            return {"X-API-Key": self._api_key}
        return {}

    async def _paginate(self, url: str) -> AsyncIterator[dict]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            cursor: str | None = None
            total = 0
            while True:
                params: dict = {"limit": _PAGE_LIMIT}
                if cursor:
                    params["cursor"] = cursor
                for attempt in range(1, 4):
                    try:
                        resp = await client.get(url, params=params, headers=self._headers())
                        break
                    except (httpx.TimeoutException, httpx.TransportError) as exc:
                        if attempt == 3:
                            raise
                        wait = 2.0 ** attempt
                        _logger.warning(
                            "Taxonomy API request failed (attempt %d/3): %s — retrying in %.0fs",
                            attempt, exc, wait,
                        )
                        await asyncio.sleep(wait)
                resp.raise_for_status()
                body = resp.json()
                items: list[dict] = body.get("data", [])
                if not items:
                    break
                for item in items:
                    yield item
                total += len(items)
                cursor = body.get("nextCursor") or None
                if not cursor:
                    break
                _logger.debug("Fetched %d items so far, advancing cursor", total)

    def fetch_occupations(self, taxonomy_model_id: str) -> AsyncIterator[dict]:
        url = f"{self._base_url}/api/app/models/{taxonomy_model_id}/occupations"
        return self._paginate(url)

    def fetch_skills(self, taxonomy_model_id: str) -> AsyncIterator[dict]:
        url = f"{self._base_url}/api/app/models/{taxonomy_model_id}/skills"
        return self._paginate(url)


class QualificationsMongoSource:
    """Reads qualifications from the nel_qualifications collection."""

    def __init__(self, app_db: AsyncIOMotorDatabase):
        self._col = app_db[Collections.NEL_QUALIFICATIONS]

    async def fetch_qualifications(self) -> AsyncIterator[dict]:
        async for doc in self._col.find({}, {"_id": 0}):
            yield doc

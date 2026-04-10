"""Dependency injection factory for IClassifyService."""

from typing import Optional

import httpx

from classify.config import NER_API_URL, NEL_API_URL
from classify.service import INERClient, INELClient, IClassifyService, ClassifyService


class _NERHttpClient(INERClient):
    async def extract(self, text: str, entity_types: Optional[list[str]] = None) -> dict:
        payload: dict = {"text": text}
        if entity_types:
            payload["entity_types"] = entity_types
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{NER_API_URL}/v1/ner", json=payload)
            resp.raise_for_status()
            return resp.json()


class _NELHttpClient(INELClient):
    async def link(self, entities: list[dict], top_k: int, min_similarity: float) -> dict:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{NEL_API_URL}/v1/nel",
                json={"entities": entities, "options": {"top_k": top_k, "min_similarity": min_similarity}},
            )
            resp.raise_for_status()
            return resp.json()


def get_classify_service() -> IClassifyService:
    return ClassifyService(ner_client=_NERHttpClient(), nel_client=_NELHttpClient())

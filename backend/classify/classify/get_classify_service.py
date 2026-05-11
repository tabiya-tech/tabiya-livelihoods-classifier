"""Dependency injection factory for IClassifyService."""

import os
from typing import Optional

import httpx
from fastapi import Depends, Request

from classify.config import NEL_API_URL, NER_API_URL
from classify.service import INERClient, INELClient, IClassifyService, ClassifyService


def build_classify_http_client() -> httpx.AsyncClient:
    """Shared AsyncClient for NER/NEL calls (connection pooling, keep-alive)."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=30),
    )


def get_shared_http_client(request: Request) -> httpx.AsyncClient:
    """FastAPI dependency: process-wide httpx client from app lifespan."""
    return request.app.state.http_client


def _gcp_identity_token(audience: str) -> str | None:
    """Fetch a GCP identity token for the given audience via the metadata server.
    Returns None when running outside GCP (e.g. local dev)."""
    try:
        resp = httpx.get(
            f"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience={audience}",
            headers={"Metadata-Flavor": "Google"},
            timeout=2.0,
        )
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


class _NERHttpClient(INERClient):
    def __init__(self, http_client: httpx.AsyncClient):
        self._client = http_client

    async def extract(self, text: str, entity_types: Optional[list[str]] = None) -> dict:
        payload: dict = {"text": text}
        if entity_types:
            payload["entity_types"] = entity_types
        headers = {}
        token = _gcp_identity_token(NER_API_URL)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        resp = await self._client.post(f"{NER_API_URL}/v1/ner", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


class _NELHttpClient(INELClient):
    def __init__(self, http_client: httpx.AsyncClient):
        self._client = http_client

    async def link(self, entities: list[dict], top_k: int, min_similarity: float) -> dict:
        headers = {}
        token = _gcp_identity_token(NEL_API_URL)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        resp = await self._client.post(
            f"{NEL_API_URL}/v1/nel",
            json={"entities": entities, "options": {"top_k": top_k, "min_similarity": min_similarity}},
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()


def get_classify_service(
    http_client: httpx.AsyncClient = Depends(get_shared_http_client),
) -> IClassifyService:
    return ClassifyService(ner_client=_NERHttpClient(http_client), nel_client=_NELHttpClient(http_client))

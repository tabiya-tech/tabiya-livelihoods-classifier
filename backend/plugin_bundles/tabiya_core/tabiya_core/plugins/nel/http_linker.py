"""HTTP-backed IEntityLinker — delegates to the legacy NEL v1 service.

The NEL plugin's Core (`core.py`) is pure business logic behind
`IEntityLinker`. In production the embedding + vector search lives in the
separate `backend/nel/` service that exposes `POST /v1/nel`. This adapter
translates the plugin's `link(entities, ...)` call into that HTTP request
and reshapes the response into the plugin contract's `Match` type.

Caveat (v1): the legacy `/v1/nel` endpoint reads its embedding model +
taxonomy from env at service startup, not per-request. The plugin's
`nel_model_id` / `taxonomy_model_id` config values are therefore
metadata-only in this deployment topology — swapping them via the editor
UI updates the pipeline document but does not change what model the
downstream service actually loads. A future subtask will expose a
per-request model override; the plugin contract is already shaped for it.
"""

from __future__ import annotations

import logging

import httpx
from tabiya_plugin_contracts import Match

from .core import EmbeddingsCacheNotReady


log = logging.getLogger("tabiya-core-bundle.nel.http-linker")


class HttpEntityLinker:
    """Calls the legacy NEL service at `{base_url}/v1/nel`."""

    def __init__(
        self,
        base_url: str,
        http_client: httpx.AsyncClient | None = None,
        request_timeout_seconds: float = 45.0,
    ) -> None:
        if not base_url:
            raise ValueError("HttpEntityLinker requires a non-empty base_url.")
        self._base_url = base_url.rstrip("/")
        self._request_timeout_seconds = request_timeout_seconds
        self._owned_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=request_timeout_seconds
        )

    async def close(self) -> None:
        """Close the internal http client if this instance owns it."""

        if self._owned_client:
            await self._http_client.aclose()

    async def link(
        self,
        entities: list[tuple[str, str]],
        nel_model_id: str,
        taxonomy_model_id: str,
        top_k: int,
        min_similarity: float,
    ) -> list[list[Match]]:
        if not entities:
            return []

        request_body = {
            "entities": [
                {"text": surface_form, "entity_type": entity_type}
                for surface_form, entity_type in entities
            ],
            "options": {"top_k": top_k, "min_similarity": min_similarity},
        }
        endpoint_url = f"{self._base_url}/v1/nel"
        log.debug(
            "Calling legacy NEL service nel_model_id=%s taxonomy=%s count=%d",
            nel_model_id,
            taxonomy_model_id,
            len(entities),
        )
        response = await self._http_client.post(endpoint_url, json=request_body)

        if response.status_code == 503:
            raise EmbeddingsCacheNotReady(
                taxonomy_model_id=taxonomy_model_id,
                nel_model_id=nel_model_id,
                current_status="upstream_unavailable",
            )
        response.raise_for_status()
        payload = response.json()

        linked_entities_raw = payload.get("linked_entities", [])
        matches_per_entity: list[list[Match]] = []
        for linked_entity in linked_entities_raw:
            raw_matches = linked_entity.get("matches", [])
            matches_per_entity.append(
                [_match_from_legacy_dict(raw_match) for raw_match in raw_matches]
            )
        return matches_per_entity


def _match_from_legacy_dict(raw_match: dict) -> Match:
    return Match(
        id=str(raw_match.get("code") or raw_match.get("uri") or raw_match.get("label", "")),
        preferred_label=raw_match["label"],
        score=float(raw_match.get("similarity_score", 0.0)),
        uri=raw_match.get("uri"),
    )


__all__ = ["HttpEntityLinker"]

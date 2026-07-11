"""HTTP-backed IEntityLinker — delegates to the NEL v2 service.

The NEL plugin's Core (`core.py`) is pure business logic behind
`IEntityLinker`. In production the embedding + vector search lives in the
separate `backend/nel_v2/` service that exposes `POST /v2/nel`. This adapter
translates the plugin's `link(entities, ...)` call into that HTTP request
and reshapes the response into the plugin contract's `Match` type.

Model selection is per-user, resolved by identity: `nel_v2` reads the
authenticated user's configured `nel_model_id` / `taxonomy_model_id` from its
user-config store. So this adapter forwards the end-user's `user_id` (as the
`x-apigateway-api-userinfo` header the service already reads) — that identity,
not the pipeline's config, decides which models `nel_v2` uses and which
(model × taxonomy) embeddings cache it queries. A GCP service-account token is
still attached for Cloud Run IAM.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Optional

import httpx
from tabiya_plugin_contracts import Match

from tabiya_core.identity import bearer_headers

from .core import EmbeddingsCacheNotReady


log = logging.getLogger("tabiya-core-bundle.nel.http-linker")


def _userinfo_header(user_id: Optional[str]) -> dict[str, str]:
    """Encode the end-user id the way the gateway would, so nel_v2's
    `get_firebase_uid` resolves this user and loads their model config."""
    if not user_id:
        return {}
    payload = base64.b64encode(json.dumps({"user_id": user_id}).encode()).decode()
    return {"x-apigateway-api-userinfo": payload}


class HttpEntityLinker:
    """Calls the NEL v2 service at `{base_url}/v2/nel`."""

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
        user_id: Optional[str] = None,
    ) -> list[list[Match]]:
        if not entities:
            return []

        # nel_v2's NELRequest puts top_k / min_similarity at the top level
        # (not under "options"). The model + taxonomy are NOT in the body —
        # nel_v2 resolves them from the authenticated user's config, so we
        # forward the user identity instead (see _userinfo_header).
        request_body = {
            "entities": [
                {"text": surface_form, "entity_type": entity_type}
                for surface_form, entity_type in entities
            ],
            "top_k": top_k,
            "min_similarity": min_similarity,
        }
        endpoint_url = f"{self._base_url}/v2/nel"
        log.debug(
            "Calling NEL v2 service user_id=%s count=%d",
            user_id,
            len(entities),
        )
        # Private Cloud Run: SA identity token for IAM (no-op off-GCP) PLUS the
        # end-user identity so nel_v2 loads *this user's* configured models.
        headers = {**bearer_headers(self._base_url), **_userinfo_header(user_id)}
        response = await self._http_client.post(endpoint_url, json=request_body, headers=headers)

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
                [_match_from_v2_dict(raw_match) for raw_match in raw_matches]
            )
        return matches_per_entity


def _match_from_v2_dict(raw_match: dict) -> Match:
    """Reshape a nel_v2 match into the plugin contract's Match.

    v2 nests the taxonomy entity under `entity` and scores as
    `similarity_score`: {entity_type, similarity_score, entity:{uuid,
    preferred_label, origin_uri, esco_code, ...}}.
    """
    entity = raw_match.get("entity", {}) or {}
    return Match(
        id=str(
            entity.get("esco_code")
            or entity.get("origin_uri")
            or entity.get("uuid")
            or entity.get("preferred_label", "")
        ),
        preferred_label=entity.get("preferred_label", ""),
        score=float(raw_match.get("similarity_score", 0.0)),
        uri=entity.get("origin_uri"),
    )


__all__ = ["HttpEntityLinker"]

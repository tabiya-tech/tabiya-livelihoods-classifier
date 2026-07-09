"""HTTP-backed IEntityExtractor — delegates to the legacy NER service.

The NER plugin's Core (`core.py`) is pure business logic behind
`IEntityExtractor`. In production the transformer model lives in the
separate `backend/ner/` service that exposes `POST /v1/ner`. This adapter
translates the plugin's `extract(text, model_id)` call into that HTTP
request and reshapes the response into the plugin contract's `Entity`
type.

Kept intentionally thin — one method, one request. No caching, no
retries: the executor + timeout live upstream.
"""

from __future__ import annotations

import logging

import httpx
from tabiya_plugin_contracts import Entity, EntitySpan


log = logging.getLogger("tabiya-core-bundle.ner.http-extractor")


class HttpEntityExtractor:
    """Calls the legacy NER service at `{base_url}/v1/ner`.

    The legacy service accepts `entity_types` server-side, but this plugin
    already filters by `entity_types` inside the Core. We deliberately do
    NOT forward `entity_types` here so the plugin owns the filtering
    contract; the extractor returns the full set the model emitted.
    """

    def __init__(
        self,
        base_url: str,
        http_client: httpx.Client | None = None,
        request_timeout_seconds: float = 30.0,
    ) -> None:
        if not base_url:
            raise ValueError("HttpEntityExtractor requires a non-empty base_url.")
        self._base_url = base_url.rstrip("/")
        self._request_timeout_seconds = request_timeout_seconds
        self._owned_client = http_client is None
        self._http_client = http_client or httpx.Client(
            timeout=request_timeout_seconds
        )

    def close(self) -> None:
        """Close the internal http client if this instance owns it."""

        if self._owned_client:
            self._http_client.close()

    def extract(self, text: str, model_id: str) -> list[Entity]:
        request_body = {"text": text}
        endpoint_url = f"{self._base_url}/v1/ner"
        log.debug(
            "Calling legacy NER service model_id=%s text_length=%d",
            model_id,
            len(text),
        )
        response = self._http_client.post(endpoint_url, json=request_body)
        response.raise_for_status()
        payload = response.json()

        raw_entities = payload.get("entities", [])
        return [_entity_from_legacy_dict(raw_entity) for raw_entity in raw_entities]


def _entity_from_legacy_dict(raw_entity: dict) -> Entity:
    raw_span = raw_entity.get("span") or {}
    return Entity(
        surface_form=raw_entity["surface_form"],
        entity_type=raw_entity["entity_type"],
        span=EntitySpan(
            start=int(raw_span.get("start", 0)),
            end=int(raw_span.get("end", 0)),
        ),
    )


__all__ = ["HttpEntityExtractor"]

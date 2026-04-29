"""Classify v2 service — orchestrates NER → NEL v2."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from classify_v2.app.classification.service.errors import (
    EmbeddingsCacheNotReadyError,
    NERServiceError,
    NELServiceError,
)
from classify_v2.app.classification.service.types import (
    ClassifiedEntity,
    ClassifyMetadata,
    ClassifyOptions,
    ClassifyResponse,
    EntitySpan,
    OccupationMatch,
    QualificationMatch,
    SkillMatch,
    TaxonomyMatch,
)
from classify_v2.config import CLASSIFIER_VERSION, NER_API_URL, NEL_V2_API_URL

_logger = logging.getLogger(__name__)


def _gcp_identity_token(audience: str) -> str | None:
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


class IClassifyService(ABC):
    @abstractmethod
    async def classify(
        self,
        input_text: str,
        options: Optional[ClassifyOptions] = None,
    ) -> ClassifyResponse: ...


class ClassifyService(IClassifyService):
    def __init__(
        self,
        ner_api_url: str = NER_API_URL,
        nel_v2_api_url: str = NEL_V2_API_URL,
    ):
        self._ner_url = ner_api_url
        self._nel_url = nel_v2_api_url

    async def classify(
        self,
        input_text: str,
        options: Optional[ClassifyOptions] = None,
    ) -> ClassifyResponse:
        opts = options or ClassifyOptions()
        start = time.monotonic()

        ner_data = await self._call_ner(input_text, opts)
        ner_entities = ner_data.get("entities", [])
        ner_model = ner_data.get("metadata", {}).get("model_name", "unknown")

        linkable_types = {"occupation", "skill", "qualification"}
        nel_input = [
            {"text": entity["surface_form"], "entity_type": entity["entity_type"]}
            for entity in ner_entities
            if entity["entity_type"] in linkable_types
        ]

        nel_response: dict = {"linked_entities": [], "metadata": {}}
        if nel_input:
            nel_response = await self._call_nel(nel_input, opts)

        linked_map: dict[tuple, list[TaxonomyMatch]] = {}
        for linked in nel_response.get("linked_entities", []):
            key = (linked["input_text"], linked["entity_type"])
            linked_map[key] = [_parse_match(match) for match in linked.get("matches", [])]

        nel_meta = nel_response.get("metadata", {})
        entities = [
            ClassifiedEntity(
                entity_type=entity["entity_type"],
                surface_form=entity["surface_form"],
                span=EntitySpan(
                    start=entity["span"]["start"],
                    end=entity["span"]["end"],
                ),
                matches=linked_map.get((entity["surface_form"], entity["entity_type"]), []),
            )
            for entity in ner_entities
        ]

        processing_time = round((time.monotonic() - start) * 1000, 1)
        return ClassifyResponse(
            entities=entities,
            metadata=ClassifyMetadata(
                classifier_version=CLASSIFIER_VERSION,
                ner_model=ner_model,
                nel_model_id=nel_meta.get("nel_model_id", "unknown"),
                taxonomy_model_id=nel_meta.get("taxonomy_model_id", "unknown"),
                processing_time_ms=processing_time,
            ),
        )

    async def _call_ner(self, input_text: str, opts: ClassifyOptions) -> dict:
        payload: dict = {"text": input_text}
        if opts.extract_entities:
            payload["entity_types"] = [e.value for e in opts.extract_entities]
        headers: dict = {}
        token = _gcp_identity_token(NER_API_URL)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{self._ner_url}/v1/ner", json=payload, headers=headers)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            raise NERServiceError(f"NER service error {exc.response.status_code}: {exc.response.text}") from exc
        except httpx.RequestError as exc:
            raise NERServiceError(f"NER service unreachable: {exc}") from exc

    async def _call_nel(self, entities: list[dict], opts: ClassifyOptions) -> dict:
        headers: dict = {}
        token = _gcp_identity_token(NEL_V2_API_URL)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self._nel_url}/v2/nel",
                    json={
                        "entities": entities,
                        "top_k": opts.top_k,
                        "min_similarity": opts.min_similarity,
                    },
                    headers=headers,
                )
                if resp.status_code == 503:
                    raise EmbeddingsCacheNotReadyError(resp.json().get("detail", "Embeddings not ready"))
                resp.raise_for_status()
                return resp.json()
        except EmbeddingsCacheNotReadyError:
            raise
        except httpx.HTTPStatusError as exc:
            raise NELServiceError(f"NEL service error {exc.response.status_code}: {exc.response.text}") from exc
        except httpx.RequestError as exc:
            raise NELServiceError(f"NEL service unreachable: {exc}") from exc


def _parse_match(raw: dict) -> TaxonomyMatch:
    entity_type = raw.get("entity_type", "occupation")
    if entity_type == "occupation":
        return OccupationMatch(**raw)
    elif entity_type == "skill":
        return SkillMatch(**raw)
    else:
        return QualificationMatch(**raw)

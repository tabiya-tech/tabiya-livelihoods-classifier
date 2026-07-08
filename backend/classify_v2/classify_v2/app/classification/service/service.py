"""Classify v2 service — orchestrates a persisted pipeline via the executor.

The legacy `_call_ner` / `_call_nel` methods are gone; every classify
request now runs whatever pipeline the caller resolves to (explicit
`pipeline_id` → user's active pipeline → seeded Default Tabiya).

Resolution order + Default Tabiya seeding both live in the classify
route, not here — this layer just takes a resolved `PipelineDocument`
and hands it to the executor. Keeping resolution out of the service
means the same executor path is reused by any future non-HTTP entry
(a background job, an MCP call, etc.) without dragging the "seed a
default" logic along.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from classify_v2.app.classification.service.errors import (
    EmbeddingsCacheNotReadyError,
    NELServiceError,
    NERServiceError,
)
from classify_v2.app.classification.service.types import (
    ClassifiedEntity,
    ClassifyMetadata,
    ClassifyOptions,
    ClassifyResponse,
    EntitySpan,
    OccupationMatch,
    PipelineStageSummary,
    PipelineSummary,
    QualificationMatch,
    SkillMatch,
    TaxonomyMatch,
)
from classify_v2.app.pipelines.executor import (
    ExecutorResult,
    PipelineExecutor,
    PluginInvocationError,
    PluginTimeoutError,
    PluginUpstreamUnavailableError,
)
from classify_v2.app.pipelines.repository import PipelineDocument
from classify_v2.config import CLASSIFIER_VERSION

_logger = logging.getLogger(__name__)


class IClassifyService(ABC):
    @abstractmethod
    async def classify(
        self,
        *,
        pipeline: PipelineDocument,
        input_text: str,
        options: Optional[ClassifyOptions] = None,
        request_id: str,
        user_id: Optional[str],
    ) -> ClassifyResponse: ...


class ClassifyService(IClassifyService):
    def __init__(self, *, executor: PipelineExecutor) -> None:
        self._executor = executor

    async def classify(
        self,
        *,
        pipeline: PipelineDocument,
        input_text: str,
        options: Optional[ClassifyOptions] = None,
        request_id: str,
        user_id: Optional[str],
    ) -> ClassifyResponse:
        source_overrides = _build_source_overrides(pipeline, input_text, options)
        started = time.monotonic()
        try:
            result = await self._executor.run(
                pipeline=pipeline,
                source_overrides=source_overrides,
                request_id=request_id,
                user_id=user_id,
            )
        except PluginUpstreamUnavailableError as exc:
            # Preserve pre-plugin classify behaviour: cache-not-ready → 503
            raise EmbeddingsCacheNotReadyError(str(exc)) from exc
        except PluginTimeoutError as exc:
            raise NELServiceError(f"Plugin timeout: {exc}") from exc
        except PluginInvocationError as exc:
            raise NERServiceError(f"Plugin invocation failed: {exc}") from exc

        processing_time_ms = round((time.monotonic() - started) * 1000, 1)
        entities = _entities_from_executor_result(result)
        nel_metadata = _collect_nel_metadata(result)
        ner_metadata = _collect_ner_metadata(result)

        return ClassifyResponse(
            entities=entities,
            metadata=ClassifyMetadata(
                classifier_version=CLASSIFIER_VERSION,
                ner_model=ner_metadata.get("model_name", "unknown"),
                nel_model_id=nel_metadata.get("nel_model_id", "unknown"),
                taxonomy_model_id=nel_metadata.get("taxonomy_model_id", "unknown"),
                processing_time_ms=processing_time_ms,
                pipeline=PipelineSummary(
                    pipeline_id=result.pipeline_id,
                    name=result.pipeline_name,
                    stages=[
                        PipelineStageSummary(
                            plugin_id=outcome.plugin_id,
                            category=outcome.category,
                        )
                        for outcome in result.stages
                    ],
                ),
            ),
        )


# ── Helpers ──────────────────────────────────────────────────────────────


_SOURCE_PLUGIN_TEXT = "tabiya.source.text.v1"


def _build_source_overrides(
    pipeline: PipelineDocument,
    input_text: str,
    options: Optional[ClassifyOptions],
) -> dict[str, Any]:
    """Turn the classify request's `input_text` (+ options) into the
    source stage's config for this run.

    For the canonical text source we drop the caller's text under `text`.
    For a future scraper source (URL input, when 11.11+ lands), we'd
    detect the source plugin and shape the override differently.
    """

    if not pipeline.stages:
        return {}
    source = pipeline.stages[0]
    if source.plugin_id == _SOURCE_PLUGIN_TEXT:
        return {"text": input_text}
    # Unknown source plugin: hand the caller's text through verbatim
    # under `text` and let the plugin's own config validator decide what
    # to do with it. That way an admin experimenting with a new source
    # doesn't need code changes here to run their first pipeline.
    return {"text": input_text}


def _entities_from_executor_result(result: ExecutorResult) -> list[ClassifiedEntity]:
    payload = result.linked_entities_payload
    if not payload:
        return []
    entities: list[ClassifiedEntity] = []
    for raw in payload.get("entities", []):
        entities.append(
            ClassifiedEntity(
                entity_type=raw.get("entity_type", ""),
                surface_form=raw.get("surface_form", ""),
                span=EntitySpan(
                    start=raw.get("span", {}).get("start", 0),
                    end=raw.get("span", {}).get("end", 0),
                ),
                matches=[_parse_match(match) for match in raw.get("matches", [])],
            )
        )
    return entities


def _parse_match(raw: dict) -> TaxonomyMatch:
    """Turn a plugin-shaped `Match` into the response's typed match model.

    The plugin contract's Match is minimal (`id`, `preferred_label`,
    `score`, `uri`), while the response shape here is the older
    `entity_type` + `similarity_score` + fully-typed `entity` — we fill
    in what we can and leave the rest as placeholders. Consumers that
    care about the richer taxonomy fields should hit the taxonomy API
    directly with the `uri` from the plugin match.
    """

    entity_type = raw.get("entity_type") or "occupation"
    identifier = raw.get("id", "")
    label = raw.get("preferred_label", "")
    uri = raw.get("uri", "") or ""
    score = float(raw.get("score", raw.get("similarity_score", 0.0)))

    base_entity_fields = {
        "uuid": identifier,
        "origin_uuid": identifier,
        "uuid_history": [],
        "preferred_label": label,
        "origin_uri": uri,
        "alt_labels": [],
        "description": "",
    }

    if entity_type == "skill":
        return SkillMatch(
            similarity_score=score,
            entity={**base_entity_fields, "skill_type": None, "reuse_level": None},
        )
    if entity_type == "qualification":
        return QualificationMatch(
            similarity_score=score,
            entity={**base_entity_fields, "eqf_level": None, "country": None},
        )
    return OccupationMatch(
        similarity_score=score,
        entity={**base_entity_fields, "esco_code": None},
    )


def _collect_ner_metadata(result: ExecutorResult) -> dict[str, Any]:
    for outcome in result.stages:
        if outcome.plugin_id == "tabiya.ner.v1" and outcome.metadata:
            return outcome.metadata
    return {}


def _collect_nel_metadata(result: ExecutorResult) -> dict[str, Any]:
    for outcome in result.stages:
        if outcome.plugin_id == "tabiya.nel.v1" and outcome.metadata:
            return outcome.metadata
    return {}

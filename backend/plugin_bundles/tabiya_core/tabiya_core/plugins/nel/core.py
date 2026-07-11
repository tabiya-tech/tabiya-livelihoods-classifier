"""NEL plugin Core — Entities → LinkedEntities.

The heavy lifting (embedding + mongo vector search) lives behind
`IEntityLinker`. That means:
  * Tests inject a deterministic fake linker.
  * The bundle wires the real MongoDB-backed linker at startup.
  * If the embeddings cache isn't ready, the linker raises
    `EmbeddingsCacheNotReady` and the adapter maps it to a 503
    `UPSTREAM_UNAVAILABLE` envelope — preserving the existing behaviour
    of the pre-plugin classify pipeline.

Linkable entity types are restricted to `occupation`, `skill`,
`qualification`. Anything else the NER stage emits (`experience`,
`domain`, ...) passes through with an empty `matches` list — the pipeline
should not error on model output that has no ESCO counterpart.
"""

from __future__ import annotations

from typing import Optional, Protocol

from pydantic import BaseModel, Field, ValidationError
from tabiya_plugin_contracts import (
    Context,
    Entities,
    LinkedEntities,
    LinkedEntity,
    Match,
)
from tabiya_plugin_contracts.adapters.http import (
    BadInputError,
    ConfigInvalidError,
    UpstreamUnavailableError,
    jsonable_validation_errors,
)


LINKABLE_ENTITY_TYPES = frozenset({"occupation", "skill", "qualification"})


class NelConfig(BaseModel):
    """Typed NEL plugin config."""

    nel_model_id: str = Field(
        ...,
        description="Embedding model to use for the query embedding.",
    )
    taxonomy_model_id: str = Field(
        ...,
        description="Taxonomy snapshot to search against.",
    )
    top_k: int = Field(default=5, ge=1, le=50)
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)


class EmbeddingsCacheNotReady(Exception):
    """Raised by the linker when the embeddings cache is not ready.

    Mapped by the plugin's `invoke` to `UpstreamUnavailableError` so the
    adapter emits a 503 `UPSTREAM_UNAVAILABLE` envelope. Kept as a distinct
    class so a future retry-with-backoff policy can catch it specifically.
    """

    def __init__(self, taxonomy_model_id: str, nel_model_id: str, current_status: str) -> None:
        super().__init__(
            f"Embeddings cache not ready for taxonomy_model={taxonomy_model_id} "
            f"nel_model={nel_model_id} (status={current_status})."
        )
        self.taxonomy_model_id = taxonomy_model_id
        self.nel_model_id = nel_model_id
        self.current_status = current_status


class IEntityLinker(Protocol):
    """Runs the embedding + vector-search side of NEL.

    Called once per invoke with the full batch of linkable entities;
    implementations are expected to embed as a batch for throughput.
    """

    async def link(
        self,
        entities: list[tuple[str, str]],
        nel_model_id: str,
        taxonomy_model_id: str,
        top_k: int,
        min_similarity: float,
        user_id: Optional[str] = None,
    ) -> list[list[Match]]: ...


_linker: Optional[IEntityLinker] = None


def set_linker(linker: IEntityLinker) -> None:
    global _linker
    _linker = linker


def get_linker() -> IEntityLinker:
    if _linker is None:
        raise RuntimeError(
            "NEL linker not initialised. Call set_linker(...) before invoke."
        )
    return _linker


async def invoke(input: Entities, config: dict, context: Context) -> LinkedEntities:
    try:
        parsed_config = NelConfig.model_validate(config or {})
    except ValidationError as exc:
        raise ConfigInvalidError(
            "NEL config failed validation.",
            detail={"errors": jsonable_validation_errors(exc)},
        ) from exc

    if not input.entities:
        return LinkedEntities(entities=[], source_text=input.source_text)

    # Split the batch into linkable vs. pass-through. Pass-through entities
    # keep their metadata but get an empty matches list — the pipeline stays
    # non-lossy for downstream Sinks.
    linkable_indices: list[int] = []
    linkable_pairs: list[tuple[str, str]] = []
    for index, entity in enumerate(input.entities):
        if entity.entity_type in LINKABLE_ENTITY_TYPES:
            linkable_indices.append(index)
            linkable_pairs.append((entity.surface_form, entity.entity_type))

    linker = get_linker()
    try:
        matches_per_entity = (
            await linker.link(
                entities=linkable_pairs,
                nel_model_id=parsed_config.nel_model_id,
                taxonomy_model_id=parsed_config.taxonomy_model_id,
                top_k=parsed_config.top_k,
                min_similarity=parsed_config.min_similarity,
                # Forward the end-user identity so the NEL service resolves
                # *this user's* configured models (per-user, via identity —
                # not the pipeline's config or a shared service account).
                user_id=context.user_id,
            )
            if linkable_pairs
            else []
        )
    except EmbeddingsCacheNotReady as exc:
        raise UpstreamUnavailableError(str(exc)) from exc

    if len(matches_per_entity) != len(linkable_pairs):
        raise BadInputError(
            "Linker returned a matches list of the wrong length.",
            detail={
                "expected": len(linkable_pairs),
                "actual": len(matches_per_entity),
            },
        )

    matches_by_index: dict[int, list[Match]] = dict(
        zip(linkable_indices, matches_per_entity)
    )

    linked_entities = [
        LinkedEntity(
            surface_form=entity.surface_form,
            entity_type=entity.entity_type,
            span=entity.span,
            matches=matches_by_index.get(index, []),
        )
        for index, entity in enumerate(input.entities)
    ]

    return LinkedEntities(entities=linked_entities, source_text=input.source_text)


__all__ = [
    "EmbeddingsCacheNotReady",
    "IEntityLinker",
    "LINKABLE_ENTITY_TYPES",
    "NelConfig",
    "get_linker",
    "invoke",
    "set_linker",
]

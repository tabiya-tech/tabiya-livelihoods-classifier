"""Embedding generation service.

Orchestrates: taxonomy source → batch embed → store in Atlas.
Tracks status in nel_embeddings_cache_status so the linker knows when it's safe
to run vector search queries.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod

import httpx
from google.api_core.exceptions import DeadlineExceeded as GoogleDeadlineExceeded
from google.api_core.exceptions import ServiceUnavailable as GoogleServiceUnavailable
from pymongo.errors import ConnectionFailure, NetworkTimeout, ExecutionTimeout

from nel.app.embedding.service.service import IEmbeddingService
from nel.app.embeddings_cache.repository.repository import IEmbeddingsCacheRepository
from nel.app.embeddings_cache.service.taxonomy_source import (
    TaxonomyAPISource,
    QualificationsMongoSource,
)
from nel.app.embeddings_cache.types import (
    CacheStatus,
    EmbeddingCacheStatus,
    EmbeddingDocument,
)
from nel.app.retry import RetryPolicy

_logger = logging.getLogger(__name__)

_ENTITY_TYPES = ["occupation", "skill", "qualification"]

# Exceptions considered transient for embedding and DB calls
_TRANSIENT = (
    asyncio.TimeoutError,
    httpx.TransportError,
    httpx.TimeoutException,
    ConnectionFailure,
    NetworkTimeout,
    ExecutionTimeout,
    GoogleDeadlineExceeded,
    GoogleServiceUnavailable,
)

_DEFAULT_RETRY = RetryPolicy(attempts=3, backoff=2.0, on=_TRANSIENT)


class IEmbeddingGenerationService(ABC):
    @abstractmethod
    async def generate_for_combination(
        self, taxonomy_model_id: str, nel_model_id: str, force: bool = False
    ) -> None: ...


class EmbeddingGenerationService(IEmbeddingGenerationService):
    def __init__(
        self,
        cache_repository: IEmbeddingsCacheRepository,
        embedding_service: IEmbeddingService,
        taxonomy_source: TaxonomyAPISource,
        qualifications_source: QualificationsMongoSource,
        page_size: int = 100,
        retry: RetryPolicy = _DEFAULT_RETRY,
    ):
        self._cache_repo = cache_repository
        self._embedding_svc = embedding_service
        self._taxonomy_source = taxonomy_source
        self._qual_source = qualifications_source
        self._batch_size = page_size
        self._retry = retry

    async def generate_for_combination(
        self,
        taxonomy_model_id: str,
        nel_model_id: str,
        force: bool = False,
        start_cursor: str | None = None,
        start_cursor_entity_type: str | None = None,
    ) -> None:
        """Generate and store embeddings for all three entity types.

        If force=True, existing embeddings are deleted and regenerated even if
        already "ready". Skips entity types that are already "ready" unless forced.

        start_cursor / start_cursor_entity_type: resume pagination from a known cursor
        for a specific entity type (e.g. after a mid-run failure). Only applied to the
        matching entity type; all others paginate from the beginning.
        """
        for entity_type in _ENTITY_TYPES:
            cursor = start_cursor if entity_type == start_cursor_entity_type else None
            await self._generate_entity_type(
                taxonomy_model_id, nel_model_id, entity_type, force, start_cursor=cursor
            )

    async def _generate_entity_type(
        self,
        taxonomy_model_id: str,
        nel_model_id: str,
        entity_type: str,
        force: bool,
        start_cursor: str | None = None,
    ) -> None:
        status = await self._cache_repo.get_cache_status(
            taxonomy_model_id, nel_model_id, entity_type
        )

        if status and status.status == CacheStatus.ready and not force:
            _logger.info(
                "Skipping %s — already ready (use force=True to regenerate)", entity_type
            )
            return

        _logger.info("Starting embedding generation: %s", entity_type)

        await self._cache_repo.upsert_cache_status(
            EmbeddingCacheStatus(
                taxonomy_model_id=taxonomy_model_id,
                nel_model_id=nel_model_id,
                entity_type=entity_type,
                status=CacheStatus.generating,
            )
        )

        # force=True means "regenerate from scratch" — wipe existing embeddings.
        if force:
            deleted = await self._cache_repo.delete_embeddings(
                taxonomy_model_id, nel_model_id, entity_type
            )
            if deleted:
                _logger.info("Deleted %d existing embeddings for %s", deleted, entity_type)

        # Item-level resume: skip items already embedded for this combination so
        # an interrupted/failed prior run continues where it left off — no
        # re-embedding (no wasted compute/Vertex cost) and no duplicates. On a
        # fresh run this is just an empty set. Skipped when forcing (we just
        # deleted everything) or resuming an explicit --start-cursor stream.
        skip_uuids: set[str] = set()
        if not force and start_cursor is None:
            skip_uuids = await self._cache_repo.get_existing_uuids(
                taxonomy_model_id, nel_model_id, entity_type
            )
            if skip_uuids:
                _logger.info(
                    "Resuming %s — %d items already embedded, will skip them",
                    entity_type,
                    len(skip_uuids),
                )

        try:
            newly_inserted = await self._run_generation(
                taxonomy_model_id,
                nel_model_id,
                entity_type,
                start_cursor=start_cursor,
                skip_uuids=skip_uuids,
            )
            # total_count reflects the full cache (items carried over from a
            # prior run + newly embedded this run), not just this run's inserts.
            total = len(skip_uuids) + newly_inserted
            await self._cache_repo.upsert_cache_status(
                EmbeddingCacheStatus(
                    taxonomy_model_id=taxonomy_model_id,
                    nel_model_id=nel_model_id,
                    entity_type=entity_type,
                    status=CacheStatus.ready,
                    total_count=total,
                )
            )
            _logger.info(
                "Done: %s — %d embeddings stored (%d new this run)",
                entity_type, total, newly_inserted,
            )

        except Exception as exc:
            _logger.exception("Failed generating %s: %s", entity_type, exc)
            await self._cache_repo.upsert_cache_status(
                EmbeddingCacheStatus(
                    taxonomy_model_id=taxonomy_model_id,
                    nel_model_id=nel_model_id,
                    entity_type=entity_type,
                    status=CacheStatus.failed,
                    error_message=str(exc),
                )
            )
            raise

    async def _run_generation(
        self,
        taxonomy_model_id: str,
        nel_model_id: str,
        entity_type: str,
        start_cursor: str | None = None,
        skip_uuids: set[str] | None = None,
    ) -> int:
        """Fetch a page, embed it, insert it, repeat. Returns total inserted count.

        Items whose UUID is in `skip_uuids` (already embedded on a prior run)
        are dropped BEFORE embedding, so a resume costs nothing for work already
        done.
        """
        skip_uuids = skip_uuids or set()
        total = 0
        skipped = 0
        t0 = time.monotonic()

        source = self._get_source(entity_type, taxonomy_model_id, start_cursor=start_cursor)
        page: list[dict] = []

        async for raw in source:
            if skip_uuids and raw.get("UUID", "") in skip_uuids:
                skipped += 1
                continue
            page.append(raw)
            if len(page) >= self._batch_size:
                total += await self._flush_page(page, entity_type, taxonomy_model_id, nel_model_id, total, t0)
                page.clear()

        if page:
            total += await self._flush_page(page, entity_type, taxonomy_model_id, nel_model_id, total, t0)

        if skipped:
            _logger.info("Skipped %d already-embedded %s item(s) on resume", skipped, entity_type)

        return total

    async def _flush_page(
        self,
        page: list[dict],
        entity_type: str,
        taxonomy_model_id: str,
        nel_model_id: str,
        running_total: int,
        t0: float,
    ) -> int:
        # Filter out items with no embeddable text — they would produce
        # meaningless zero-vector embeddings and pollute search results.
        valid = [(raw, _embedded_text(raw, entity_type)) for raw in page]
        skipped = [(raw, text) for raw, text in valid if not text.strip()]
        valid = [(raw, text) for raw, text in valid if text.strip()]

        if skipped:
            _logger.warning(
                "[%s] Skipping %d item(s) with empty label: uuids=%s",
                entity_type,
                len(skipped),
                [r.get("UUID", r.get("qualification", "?")) for r, _ in skipped],
            )

        if not valid:
            return 0

        raws, texts = zip(*valid)

        embeddings = await self._retry.run(
            lambda: self._embedding_svc.embed_batch(list(texts))
        )
        docs = [
            _build_document(raw, emb, text, entity_type, taxonomy_model_id, nel_model_id)
            for raw, emb, text in zip(raws, embeddings, texts)
        ]
        inserted = await self._retry.run(
            lambda: self._cache_repo.insert_embeddings_batch(entity_type, docs)
        )
        elapsed = time.monotonic() - t0
        _logger.info("[%s] %d embedded so far (%.1fs)", entity_type, running_total + inserted, elapsed)
        return inserted

    def _get_source(self, entity_type: str, taxonomy_model_id: str, start_cursor: str | None = None):
        if entity_type == "occupation":
            return self._taxonomy_source.fetch_occupations(taxonomy_model_id, start_cursor=start_cursor)
        elif entity_type == "skill":
            return self._taxonomy_source.fetch_skills(taxonomy_model_id, start_cursor=start_cursor)
        else:
            return self._qual_source.fetch_qualifications()


def _embedded_text(raw: dict, entity_type: str) -> str:
    """The text that gets embedded — preferred label for occ/skill, qualification name for qual."""
    if entity_type == "qualification":
        return raw.get("qualification", "")
    return raw.get("preferredLabel", "")


def _build_document(
    raw: dict,
    embedding: list[float],
    embedded_text: str,
    entity_type: str,
    taxonomy_model_id: str,
    nel_model_id: str,
) -> EmbeddingDocument:
    if entity_type == "qualification":
        return EmbeddingDocument(
            taxonomy_model_id=taxonomy_model_id,
            nel_model_id=nel_model_id,
            entity_type=entity_type,
            entity_uuid=raw.get("UUID", ""),
            origin_uuid=raw.get("originUUID", raw.get("UUID", "")),
            uuid_history=raw.get("UUIDHistory", []),
            preferred_label=raw.get("qualification", ""),
            origin_uri=raw.get("originUri", ""),
            alt_labels=raw.get("altLabels", []),
            description=raw.get("description", ""),
            eqf_level=str(raw.get("eqf_level", "")) if raw.get("eqf_level") is not None else None,
            country=raw.get("country"),
            embedded_text=embedded_text,
            embedding=embedding,
        )
    elif entity_type == "occupation":
        return EmbeddingDocument(
            taxonomy_model_id=taxonomy_model_id,
            nel_model_id=nel_model_id,
            entity_type=entity_type,
            entity_uuid=raw.get("UUID", ""),
            origin_uuid=raw.get("originUUID", ""),
            uuid_history=raw.get("UUIDHistory", []),
            preferred_label=raw.get("preferredLabel", ""),
            origin_uri=raw.get("originUri", ""),
            alt_labels=raw.get("altLabels", []),
            description=raw.get("description", ""),
            esco_code=raw.get("code"),
            embedded_text=embedded_text,
            embedding=embedding,
        )
    else:  # skill
        return EmbeddingDocument(
            taxonomy_model_id=taxonomy_model_id,
            nel_model_id=nel_model_id,
            entity_type=entity_type,
            entity_uuid=raw.get("UUID", ""),
            origin_uuid=raw.get("originUUID", ""),
            uuid_history=raw.get("UUIDHistory", []),
            preferred_label=raw.get("preferredLabel", ""),
            origin_uri=raw.get("originUri", ""),
            alt_labels=raw.get("altLabels", []),
            description=raw.get("description", ""),
            skill_type=raw.get("skillType"),
            reuse_level=raw.get("reuseLevel"),
            embedded_text=embedded_text,
            embedding=embedding,
        )

"""Entity linking service."""

import logging
import time
from abc import ABC, abstractmethod

from nel.app.embeddings_cache.repository.repository import IEmbeddingsCacheRepository
from nel.app.embeddings_cache.types import CacheStatus
from nel.app.embedding.service.service import IEmbeddingService
from nel.app.linking.repository.repository import IEntityLinkingRepository
from nel.app.linking.service.errors import EmbeddingsCacheNotReadyError
from nel.app.linking.service.types import (
    EntityType,
    LinkedEntity,
    NELMetadata,
    NELOptions,
    NELResponse,
)

_logger = logging.getLogger(__name__)

_ENTITY_TYPES = [e.value for e in EntityType]


class INELService(ABC):
    @abstractmethod
    async def link(
        self,
        entities: list[tuple[str, EntityType]],
        taxonomy_model_id: str,
        nel_model_id: str,
        options: NELOptions,
    ) -> NELResponse: ...


class NELService(INELService):
    def __init__(
        self,
        linking_repository: IEntityLinkingRepository,
        cache_repository: IEmbeddingsCacheRepository,
        embedding_service: IEmbeddingService,
    ):
        self._linking_repo = linking_repository
        self._cache_repo = cache_repository
        self._embedding_svc = embedding_service

    async def link(
        self,
        entities: list[tuple[str, EntityType]],
        taxonomy_model_id: str,
        nel_model_id: str,
        options: NELOptions,
    ) -> NELResponse:
        # Check cache readiness for each distinct entity type requested
        for entity_type in {et for _, et in entities}:
            status = await self._cache_repo.get_cache_status(
                taxonomy_model_id, nel_model_id, entity_type.value
            )
            if status is None or status.status != CacheStatus.ready:
                current = status.status.value if status else "not found"
                raise EmbeddingsCacheNotReadyError(taxonomy_model_id, nel_model_id, current)

        _logger.debug(
            "Linking %d entities | taxonomy_model=%s nel_model=%s top_k=%d min_similarity=%.2f",
            len(entities), taxonomy_model_id, nel_model_id, options.top_k, options.min_similarity,
        )

        t0 = time.monotonic()
        texts = [text for text, _ in entities]
        embeddings = await self._embedding_svc.embed_batch(texts)

        linked = []
        for (text, entity_type), embedding in zip(entities, embeddings):
            matches = await self._linking_repo.find_matches(
                entity_type=entity_type.value,
                query_embedding=embedding,
                taxonomy_model_id=taxonomy_model_id,
                nel_model_id=nel_model_id,
                top_k=options.top_k,
                min_similarity=options.min_similarity,
            )
            _logger.debug(
                "  [%s] %r → %d match(es)%s",
                entity_type.value,
                text,
                len(matches),
                f" (top: {matches[0].entity.preferred_label!r} score={matches[0].similarity_score:.3f})" if matches else "",
            )
            linked.append(LinkedEntity(
                input_text=text,
                entity_type=entity_type,
                matches=matches,
            ))

        processing_time_ms = (time.monotonic() - t0) * 1000
        _logger.debug("Linking complete in %.1f ms", processing_time_ms)
        return NELResponse(
            linked_entities=linked,
            metadata=NELMetadata(
                nel_model_id=nel_model_id,
                taxonomy_model_id=taxonomy_model_id,
                processing_time_ms=processing_time_ms,
            ),
        )

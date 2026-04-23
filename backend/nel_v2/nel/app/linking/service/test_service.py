"""Tests for NELService."""

import pytest
from unittest.mock import AsyncMock

from nel.app.embeddings_cache.types import CacheStatus, EmbeddingCacheStatus
from nel.app.linking.repository.repository import IEntityLinkingRepository
from nel.app.linking.service.errors import EmbeddingsCacheNotReadyError
from nel.app.linking.service.service import NELService
from nel.app.linking.service.types import EntityType, NELOptions, OccupationMatch, OccupationEntity
from nel.app.embedding.service.service import IEmbeddingService


class FakeEmbeddingService(IEmbeddingService):
    model_id = "fake"
    dimensions = 4

    async def embed(self, text):
        return [0.1, 0.2, 0.3, 0.4]

    async def embed_batch(self, texts):
        return [[0.1, 0.2, 0.3, 0.4]] * len(texts)


def _ready_status(entity_type: str) -> EmbeddingCacheStatus:
    return EmbeddingCacheStatus(
        taxonomy_model_id="tax-1", nel_model_id="nel-1",
        entity_type=entity_type, status=CacheStatus.ready, total_count=10,
    )


def _make_match(label: str) -> OccupationMatch:
    return OccupationMatch(
        similarity_score=0.9,
        entity=OccupationEntity(
            uuid="u1", origin_uuid="u1", uuid_history=["u1"],
            preferred_label=label, origin_uri="http://example.com",
            alt_labels=[], description="",
        ),
    )


class TestNELService:
    def _make_svc(self, cache_status=None, matches=None):
        cache_repo = AsyncMock()
        cache_repo.get_cache_status.return_value = cache_status
        linking_repo = AsyncMock(spec=IEntityLinkingRepository)
        linking_repo.find_matches.return_value = matches or []
        return NELService(linking_repo, cache_repo, FakeEmbeddingService()), linking_repo, cache_repo

    async def test_happy_path_returns_linked_entities(self):
        # GIVEN cache is ready and linking returns a match
        svc, linking_repo, _ = self._make_svc(
            cache_status=_ready_status("occupation"),
            matches=[_make_match("Head Chef")],
        )

        # WHEN link is called
        result = await svc.link(
            entities=[("Head Chef", EntityType.occupation)],
            taxonomy_model_id="tax-1",
            nel_model_id="nel-1",
            options=NELOptions(),
        )

        # THEN response contains linked entity with match
        assert len(result.linked_entities) == 1
        assert result.linked_entities[0].input_text == "Head Chef"
        assert result.linked_entities[0].matches[0].entity.preferred_label == "Head Chef"
        assert result.metadata.nel_model_id == "nel-1"

    async def test_raises_when_cache_not_ready(self):
        # GIVEN cache status is "generating"
        generating_status = EmbeddingCacheStatus(
            taxonomy_model_id="tax-1", nel_model_id="nel-1",
            entity_type="occupation", status=CacheStatus.generating,
        )
        svc, _, _ = self._make_svc(cache_status=generating_status)

        # WHEN link is called
        with pytest.raises(EmbeddingsCacheNotReadyError) as exc_info:
            await svc.link(
                entities=[("Head Chef", EntityType.occupation)],
                taxonomy_model_id="tax-1",
                nel_model_id="nel-1",
                options=NELOptions(),
            )

        assert exc_info.value.status == "generating"

    async def test_raises_when_cache_status_missing(self):
        # GIVEN no cache status exists
        svc, _, _ = self._make_svc(cache_status=None)

        with pytest.raises(EmbeddingsCacheNotReadyError) as exc_info:
            await svc.link(
                entities=[("Python", EntityType.skill)],
                taxonomy_model_id="tax-1",
                nel_model_id="nel-1",
                options=NELOptions(),
            )

        assert exc_info.value.status == "not found"

    async def test_multiple_entities_different_types(self):
        # GIVEN both occupation and skill are ready
        cache_repo = AsyncMock()
        cache_repo.get_cache_status.side_effect = lambda t, n, et: _ready_status(et)
        linking_repo = AsyncMock(spec=IEntityLinkingRepository)
        linking_repo.find_matches.return_value = []
        svc = NELService(linking_repo, cache_repo, FakeEmbeddingService())

        result = await svc.link(
            entities=[("Head Chef", EntityType.occupation), ("Python", EntityType.skill)],
            taxonomy_model_id="tax-1",
            nel_model_id="nel-1",
            options=NELOptions(),
        )

        assert len(result.linked_entities) == 2
        assert linking_repo.find_matches.call_count == 2

"""Tests for EmbeddingGenerationService."""

from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from nel.app.embeddings_cache.repository.repository import IEmbeddingsCacheRepository
from nel.app.embeddings_cache.service.generation_service import EmbeddingGenerationService
from nel.app.embeddings_cache.types import CacheStatus, EmbeddingCacheStatus, EmbeddingDocument
from nel.app.embedding.service.service import IEmbeddingService
from nel.app.retry import RetryPolicy


# ── Fakes ─────────────────────────────────────────────────────────────────

class FakeEmbeddingService(IEmbeddingService):
    model_id = "fake-model"
    dimensions = 4

    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4]] * len(texts)


class FakeCacheRepository(IEmbeddingsCacheRepository):
    def __init__(self):
        self.statuses: dict[tuple, EmbeddingCacheStatus] = {}
        self.inserted: dict[str, list[EmbeddingDocument]] = {}
        self.deleted: dict[tuple, int] = {}

    async def get_cache_status(self, taxonomy_model_id, nel_model_id, entity_type):
        return self.statuses.get((taxonomy_model_id, nel_model_id, entity_type))

    async def upsert_cache_status(self, status: EmbeddingCacheStatus):
        self.statuses[(status.taxonomy_model_id, status.nel_model_id, status.entity_type)] = status

    async def delete_embeddings(self, taxonomy_model_id, nel_model_id, entity_type):
        key = (taxonomy_model_id, nel_model_id, entity_type)
        count = len(self.inserted.get(entity_type, []))
        self.inserted[entity_type] = []
        self.deleted[key] = count
        return count

    async def insert_embeddings_batch(self, entity_type, documents) -> int:
        self.inserted.setdefault(entity_type, []).extend(documents)
        return len(documents)

    async def vector_search(self, **_kwargs):
        return []

    async def ensure_indexes(self):
        pass


def _fake_taxonomy_source(occupations=None, skills=None):
    """Returns a fake TaxonomyAPISource with configurable data."""
    source = MagicMock()

    async def _occ_gen(_id):
        for item in (occupations or []):
            yield item

    async def _skill_gen(_id):
        for item in (skills or []):
            yield item

    source.fetch_occupations.side_effect = _occ_gen
    source.fetch_skills.side_effect = _skill_gen
    return source


def _fake_qual_source(qualifications=None):
    source = MagicMock()

    async def _qual_gen():
        for item in (qualifications or []):
            yield item

    source.fetch_qualifications.side_effect = _qual_gen
    return source


def _occ(uuid: str) -> dict:
    return {
        "UUID": uuid, "originUUID": uuid, "UUIDHistory": [uuid],
        "preferredLabel": f"Occupation {uuid}", "originUri": f"http://example.com/{uuid}",
        "altLabels": [], "description": "desc", "code": "1234",
    }


def _skill(uuid: str) -> dict:
    return {
        "UUID": uuid, "originUUID": uuid, "UUIDHistory": [uuid],
        "preferredLabel": f"Skill {uuid}", "originUri": f"http://example.com/{uuid}",
        "altLabels": [], "description": "desc", "skillType": "skill/competence", "reuseLevel": "cross-sector",
    }


def _qual(name: str) -> dict:
    return {"qualification": name, "country": "XX", "eqf_level": "6"}


# ── Tests ─────────────────────────────────────────────────────────────────

class TestEmbeddingGenerationService:
    _no_retry = RetryPolicy(attempts=1)

    def _make_svc(self, occupations=None, skills=None, qualifications=None, page_size=100, embedding_svc=None):
        repo = FakeCacheRepository()
        svc = EmbeddingGenerationService(
            cache_repository=repo,
            embedding_service=embedding_svc or FakeEmbeddingService(),
            taxonomy_source=_fake_taxonomy_source(occupations, skills),
            qualifications_source=_fake_qual_source(qualifications),
            page_size=page_size,
            retry=self._no_retry,
        )
        return svc, repo

    async def test_generates_all_three_entity_types(self):
        # GIVEN data for all three entity types
        svc, repo = self._make_svc(
            occupations=[_occ("o1")],
            skills=[_skill("s1")],
            qualifications=[_qual("Bachelor")],
        )

        # WHEN generate_for_combination is called
        await svc.generate_for_combination("tax-1", "nel-1")

        # THEN all three types have "ready" status
        for entity_type in ["occupation", "skill", "qualification"]:
            status = repo.statuses[("tax-1", "nel-1", entity_type)]
            assert status.status == CacheStatus.ready
            assert status.total_count == 1

    async def test_skips_already_ready_entity_types(self):
        # GIVEN occupation is already "ready"
        svc, repo = self._make_svc(occupations=[_occ("o1")])
        repo.statuses[("tax-1", "nel-1", "occupation")] = EmbeddingCacheStatus(
            taxonomy_model_id="tax-1", nel_model_id="nel-1",
            entity_type="occupation", status=CacheStatus.ready, total_count=99,
        )

        # WHEN generate_for_combination is called without force
        await svc.generate_for_combination("tax-1", "nel-1", force=False)

        # THEN occupation is NOT re-generated (count stays 99)
        assert repo.statuses[("tax-1", "nel-1", "occupation")].total_count == 99
        # AND occupation embeddings were not inserted
        assert "occupation" not in repo.inserted or len(repo.inserted.get("occupation", [])) == 0

    async def test_force_regenerates_ready_entity_types(self):
        # GIVEN occupation is already "ready" with 99 items
        svc, repo = self._make_svc(occupations=[_occ("o1"), _occ("o2")])
        repo.statuses[("tax-1", "nel-1", "occupation")] = EmbeddingCacheStatus(
            taxonomy_model_id="tax-1", nel_model_id="nel-1",
            entity_type="occupation", status=CacheStatus.ready, total_count=99,
        )

        # WHEN generate_for_combination is called with force=True
        await svc.generate_for_combination("tax-1", "nel-1", force=True)

        # THEN occupation is regenerated with new count
        assert repo.statuses[("tax-1", "nel-1", "occupation")].total_count == 2

    async def test_batching_inserts_multiple_batches(self):
        # GIVEN 5 occupations and page_size=2
        occs = [_occ(f"o{i}") for i in range(5)]
        svc, repo = self._make_svc(occupations=occs, page_size=2)

        # WHEN generated
        await svc.generate_for_combination("tax-1", "nel-1")

        # THEN all 5 were inserted (across multiple batch calls)
        assert len(repo.inserted.get("occupation", [])) == 5

    async def test_sets_failed_status_on_error(self):
        # GIVEN the taxonomy source raises an error
        source = MagicMock()
        async def _fail(_id):
            raise RuntimeError("API down")
            yield  # makes it a generator
        source.fetch_occupations.side_effect = _fail

        repo = FakeCacheRepository()
        svc = EmbeddingGenerationService(
            cache_repository=repo,
            embedding_service=FakeEmbeddingService(),
            taxonomy_source=source,
            qualifications_source=_fake_qual_source(),
        )

        # WHEN generate_for_combination is called
        with pytest.raises(RuntimeError, match="API down"):
            await svc.generate_for_combination("tax-1", "nel-1")

        # THEN occupation status is "failed"
        status = repo.statuses.get(("tax-1", "nel-1", "occupation"))
        assert status is not None
        assert status.status == CacheStatus.failed
        assert "API down" in status.error_message

    async def test_skips_items_with_empty_label(self):
        # GIVEN one occupation with a label and one without
        occs = [_occ("o1"), {**_occ("o2"), "preferredLabel": ""}]
        svc, repo = self._make_svc(occupations=occs)

        # WHEN generated
        await svc.generate_for_combination("tax-1", "nel-1")

        # THEN only the item with a label was inserted
        assert len(repo.inserted.get("occupation", [])) == 1
        assert repo.inserted["occupation"][0].entity_uuid == "o1"

    async def test_skips_items_with_whitespace_only_label(self):
        # GIVEN an occupation whose label is only whitespace
        occs = [_occ("o1"), {**_occ("o2"), "preferredLabel": "   "}]
        svc, repo = self._make_svc(occupations=occs)

        await svc.generate_for_combination("tax-1", "nel-1")

        assert len(repo.inserted.get("occupation", [])) == 1

    async def test_retry_policy_retries_on_transient_embed_error(self):
        # GIVEN an embedding service that fails twice then succeeds
        call_count = 0

        class FlakyEmbeddingService(IEmbeddingService):
            model_id = "flaky"
            dimensions = 4

            async def embed(self, text):
                return [0.1, 0.2, 0.3, 0.4]

            async def embed_batch(self, texts):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    import httpx
                    raise httpx.TransportError("network blip")
                return [[0.1, 0.2, 0.3, 0.4]] * len(texts)

        repo = FakeCacheRepository()
        svc = EmbeddingGenerationService(
            cache_repository=repo,
            embedding_service=FlakyEmbeddingService(),
            taxonomy_source=_fake_taxonomy_source(occupations=[_occ("o1")]),
            qualifications_source=_fake_qual_source(),
            retry=RetryPolicy(attempts=3, backoff=0.0, on=(Exception,)),
        )

        # WHEN generated
        await svc.generate_for_combination("tax-1", "nel-1")

        # THEN it succeeded after retries
        assert call_count == 3
        assert len(repo.inserted.get("occupation", [])) == 1

    async def test_retry_policy_raises_after_all_attempts_exhausted(self):
        # GIVEN an embedding service that always fails with a transient error
        class AlwaysFailEmbeddingService(IEmbeddingService):
            model_id = "always-fail"
            dimensions = 4

            async def embed(self, text):
                raise httpx.TransportError("always down")

            async def embed_batch(self, texts):
                import httpx
                raise httpx.TransportError("always down")

        repo = FakeCacheRepository()
        svc = EmbeddingGenerationService(
            cache_repository=repo,
            embedding_service=AlwaysFailEmbeddingService(),
            taxonomy_source=_fake_taxonomy_source(occupations=[_occ("o1")]),
            qualifications_source=_fake_qual_source(),
            retry=RetryPolicy(attempts=2, backoff=0.0, on=(Exception,)),
        )

        # WHEN generated
        import httpx
        with pytest.raises(httpx.TransportError):
            await svc.generate_for_combination("tax-1", "nel-1")

        # THEN status is failed
        status = repo.statuses.get(("tax-1", "nel-1", "occupation"))
        assert status.status == CacheStatus.failed

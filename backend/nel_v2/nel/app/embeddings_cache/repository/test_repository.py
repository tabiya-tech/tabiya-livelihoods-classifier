"""Tests for EmbeddingsCacheRepository.

Vector search tests assert pipeline shape only (in-memory MongoDB doesn't
support $vectorSearch). All other operations test against real in-memory MongoDB.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nel.app.embeddings_cache.repository.repository import EmbeddingsCacheRepository
from nel.app.embeddings_cache.types import CacheStatus, EmbeddingCacheStatus, EmbeddingDocument
from nel.conftest import patch_db_provider


def _make_doc(entity_type="occupation", uuid="uuid-1") -> EmbeddingDocument:
    return EmbeddingDocument(
        taxonomy_model_id="tax-1",
        nel_model_id="nel-1",
        entity_type=entity_type,
        entity_uuid=uuid,
        origin_uuid=uuid,
        uuid_history=[uuid],
        preferred_label="Test Label",
        origin_uri="http://example.com",
        alt_labels=["alt1"],
        description="A description",
        embedded_text="Test Label",
        embedding=[0.1] * 384,
    )


@pytest.fixture
async def repo(in_memory_mongo_server, in_memory_application_db_name, in_memory_taxonomy_db_name, in_memory_application_database, in_memory_taxonomy_database):
    r = EmbeddingsCacheRepository(
        app_db=in_memory_application_database,
        taxonomy_db=in_memory_taxonomy_database,
    )
    await r.ensure_indexes()
    return r


class TestCacheStatus:
    async def test_get_returns_none_when_not_exists(self, repo):
        # GIVEN no status exists
        # WHEN get_cache_status is called
        result = await repo.get_cache_status("tax-1", "nel-1", "occupation")
        # THEN None is returned
        assert result is None

    async def test_upsert_then_get(self, repo):
        # GIVEN a status is upserted
        status = EmbeddingCacheStatus(
            taxonomy_model_id="tax-1",
            nel_model_id="nel-1",
            entity_type="occupation",
            status=CacheStatus.generating,
        )
        await repo.upsert_cache_status(status)

        # WHEN get_cache_status is called
        result = await repo.get_cache_status("tax-1", "nel-1", "occupation")

        # THEN the correct status is returned
        assert result is not None
        assert result.status == CacheStatus.generating
        assert result.taxonomy_model_id == "tax-1"

    async def test_upsert_updates_existing(self, repo):
        # GIVEN a status is first set to generating
        status = EmbeddingCacheStatus(
            taxonomy_model_id="tax-1", nel_model_id="nel-1", entity_type="occupation",
            status=CacheStatus.generating,
        )
        await repo.upsert_cache_status(status)

        # WHEN it is updated to ready with a count
        status.status = CacheStatus.ready
        status.total_count = 42
        await repo.upsert_cache_status(status)

        # THEN the updated values are returned
        result = await repo.get_cache_status("tax-1", "nel-1", "occupation")
        assert result.status == CacheStatus.ready
        assert result.total_count == 42


class TestInsertAndDelete:
    async def test_insert_batch_and_delete(self, repo):
        # GIVEN two embedding documents
        docs = [_make_doc(uuid="uuid-1"), _make_doc(uuid="uuid-2")]

        # WHEN they are inserted
        inserted = await repo.insert_embeddings_batch("occupation", docs)
        assert inserted == 2

        # THEN they can be found in the collection
        count = await repo._embedding_col("occupation").count_documents(
            {"taxonomy_model_id": "tax-1", "nel_model_id": "nel-1"}
        )
        assert count == 2

        # WHEN they are deleted
        deleted = await repo.delete_embeddings("tax-1", "nel-1", "occupation")

        # THEN the count matches and the collection is empty
        assert deleted == 2
        count_after = await repo._embedding_col("occupation").count_documents({})
        assert count_after == 0

    async def test_insert_empty_batch_is_noop(self, repo):
        # WHEN an empty batch is inserted
        await repo.insert_embeddings_batch("occupation", [])
        # THEN nothing fails and collection is empty
        count = await repo._embedding_col("occupation").count_documents({})
        assert count == 0


class TestVectorSearchPipelineShape:
    async def test_vector_search_builds_correct_pipeline(self, repo):
        # GIVEN a mocked collection aggregate
        mock_col = MagicMock()
        mock_col.aggregate = MagicMock(return_value=_async_iter([]))

        with patch.object(repo, "_embedding_col", return_value=mock_col):
            await repo.vector_search(
                entity_type="occupation",
                query_embedding=[0.1] * 384,
                taxonomy_model_id="tax-1",
                nel_model_id="nel-1",
                top_k=5,
                min_similarity=0.5,
                index_name="nel_embedding_index_384",
            )

        # THEN aggregate was called with a $vectorSearch stage
        call_args = mock_col.aggregate.call_args
        pipeline = call_args.args[0]
        assert pipeline[0]["$vectorSearch"]["index"] == "nel_embedding_index_384"
        assert pipeline[0]["$vectorSearch"]["path"] == "embedding"
        assert pipeline[0]["$vectorSearch"]["limit"] == 5
        assert pipeline[0]["$vectorSearch"]["numCandidates"] == 150
        filter_ = pipeline[0]["$vectorSearch"]["filter"]
        assert filter_["taxonomy_model_id"]["$eq"] == "tax-1"
        assert filter_["nel_model_id"]["$eq"] == "nel-1"

    async def test_num_candidates_is_stable_across_top_k_values(self, repo):
        mock_col = MagicMock()
        mock_col.aggregate = MagicMock(return_value=_async_iter([]))

        # WHEN vector_search is called with different top_k values
        observed_num_candidates = []
        for top_k in [1, 3, 5, 10, 20]:
            mock_col.aggregate.reset_mock()
            mock_col.aggregate.return_value = _async_iter([])
            with patch.object(repo, "_embedding_col", return_value=mock_col):
                await repo.vector_search(
                    entity_type="occupation",
                    query_embedding=[0.1] * 384,
                    taxonomy_model_id="tax-1",
                    nel_model_id="nel-1",
                    top_k=top_k,
                    min_similarity=0.0,
                    index_name="nel_embedding_index_384",
                )
            pipeline = mock_col.aggregate.call_args.args[0]
            observed_num_candidates.append(pipeline[0]["$vectorSearch"]["numCandidates"])

        # THEN numCandidates is the same regardless of top_k
        assert len(set(observed_num_candidates)) == 1, (
            f"numCandidates varied across top_k values: {observed_num_candidates}"
        )


def _async_iter(items):
    """Helper: return an async iterator over items."""
    async def _gen():
        for item in items:
            yield item
    return _gen()

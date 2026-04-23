"""Tests for EntityLinkingRepository."""

from unittest.mock import AsyncMock

import pytest

from nel.app.embeddings_cache.types import EmbeddingDocument
from nel.app.linking.repository.repository import EntityLinkingRepository


def _make_doc(**kwargs) -> EmbeddingDocument:
    defaults = dict(
        taxonomy_model_id="tax-1", nel_model_id="nel-1", entity_type="occupation",
        entity_uuid="uuid-1", origin_uuid="uuid-1", uuid_history=["uuid-1"],
        preferred_label="Head Chef", origin_uri="http://example.com",
        alt_labels=[], description="Manages kitchen", esco_code="1234",
        embedded_text="Head Chef", embedding=[0.1] * 384,
    )
    defaults.update(kwargs)
    return EmbeddingDocument(**defaults)


class TestEntityLinkingRepository:
    async def test_returns_mapped_taxonomy_matches(self):
        # GIVEN cache repo returns one result
        doc = _make_doc()
        mock_cache = AsyncMock()
        mock_cache.vector_search.return_value = [(doc, 0.92)]

        repo = EntityLinkingRepository(mock_cache)

        # WHEN find_matches is called
        results = await repo.find_matches(
            entity_type="occupation",
            query_embedding=[0.1] * 384,
            taxonomy_model_id="tax-1",
            nel_model_id="nel-1",
            top_k=5,
            min_similarity=0.0,
        )

        # THEN result is an OccupationMatch with correct nested fields
        assert len(results) == 1
        match = results[0]
        assert match.similarity_score == 0.92
        assert match.entity_type.value == "occupation"
        assert match.entity.preferred_label == "Head Chef"
        assert match.entity.esco_code == "1234"
        assert match.entity.uuid == "uuid-1"

    async def test_passes_correct_args_to_vector_search(self):
        # GIVEN
        mock_cache = AsyncMock()
        mock_cache.vector_search.return_value = []
        repo = EntityLinkingRepository(mock_cache)

        # WHEN
        await repo.find_matches(
            entity_type="skill",
            query_embedding=[0.5] * 384,
            taxonomy_model_id="tax-1",
            nel_model_id="nel-1",
            top_k=10,
            min_similarity=0.7,
        )

        # THEN vector_search called with correct args
        mock_cache.vector_search.assert_called_once_with(
            entity_type="skill",
            query_embedding=[0.5] * 384,
            taxonomy_model_id="tax-1",
            nel_model_id="nel-1",
            top_k=10,
            min_similarity=0.7,
            index_name="nel_embedding_index_384",
        )

    async def test_returns_empty_when_no_results(self):
        mock_cache = AsyncMock()
        mock_cache.vector_search.return_value = []
        repo = EntityLinkingRepository(mock_cache)

        results = await repo.find_matches("occupation", [0.1] * 384, "t", "n", 5, 0.0)
        assert results == []

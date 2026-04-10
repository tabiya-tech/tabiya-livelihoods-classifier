"""Tests for ClassifyService."""

from unittest.mock import AsyncMock

import pytest

from classify.models import ClassifyOptions
from classify.service import ClassifyService, INERClient, INELClient


def _make_ner_response(entities: list[dict] | None = None) -> dict:
    return {
        "entities": entities or [
            {"entity_type": "occupation", "surface_form": "Head Chef", "span": {"start": 9, "end": 18}},
            {"entity_type": "skill", "surface_form": "plan menus", "span": {"start": 33, "end": 43}},
        ],
        "metadata": {"model_name": "tabiya/roberta-base-job-ner", "entity_count": 2, "processing_time_ms": 42.0},
    }


def _make_nel_response(linked_entities: list[dict] | None = None) -> dict:
    return {
        "linked_entities": linked_entities or [
            {
                "input_text": "Head Chef",
                "entity_type": "occupation",
                "matches": [{"similarity_score": 0.91, "taxonomy": "esco", "label": "chef"}],
            }
        ],
        "metadata": {"linker_model": "all-MiniLM-L6-v2", "taxonomy": "esco", "processing_time_ms": 18.0},
    }


class MockNERClient(INERClient):
    async def extract(self, text, entity_types=None):
        raise NotImplementedError()


class MockNELClient(INELClient):
    async def link(self, entities, top_k, min_similarity):
        raise NotImplementedError()


class TestClassifyService:
    class TestClassify:
        @pytest.mark.asyncio
        async def test_classify_successful(self):
            # GIVEN mock NER and NEL clients that return valid responses
            mock_ner = MockNERClient()
            mock_ner.extract = AsyncMock(return_value=_make_ner_response())
            mock_nel = MockNELClient()
            mock_nel.link = AsyncMock(return_value=_make_nel_response())
            service = ClassifyService(ner_client=mock_ner, nel_client=mock_nel)

            # WHEN classifying a job description
            result = await service.classify("We need a Head Chef who can plan menus.")

            # THEN two entities are returned with the occupation linked
            assert len(result.classification.entities) == 2
            assert result.classification.entity_counts == {"occupation": 1, "skill": 1}
            assert result.metadata.model_name == "tabiya/roberta-base-job-ner"
            assert result.metadata.linker_model == "all-MiniLM-L6-v2"

            # AND NER was called with the text
            mock_ner.extract.assert_called_once_with("We need a Head Chef who can plan menus.", None)

            # AND NEL was called only for linkable entity types (occupation, skill)
            mock_nel.link.assert_called_once()
            nel_call_entities = mock_nel.link.call_args[0][0]
            assert len(nel_call_entities) == 2

        @pytest.mark.asyncio
        async def test_classify_skips_nel_when_no_linkable_entities(self):
            # GIVEN NER returns only experience entities (not linkable)
            mock_ner = MockNERClient()
            mock_ner.extract = AsyncMock(return_value=_make_ner_response(entities=[
                {"entity_type": "experience", "surface_form": "5 years", "span": {"start": 0, "end": 7}},
            ]))
            mock_nel = MockNELClient()
            mock_nel.link = AsyncMock()
            service = ClassifyService(ner_client=mock_ner, nel_client=mock_nel)

            # WHEN classifying
            result = await service.classify("5 years experience required.")

            # THEN NEL is never called
            mock_nel.link.assert_not_called()

            # AND one entity is returned without linked_entities
            assert len(result.classification.entities) == 1
            assert result.classification.entities[0].linked_entities is None

        @pytest.mark.asyncio
        async def test_classify_passes_options_to_clients(self):
            # GIVEN mock clients
            mock_ner = MockNERClient()
            mock_ner.extract = AsyncMock(return_value=_make_ner_response())
            mock_nel = MockNELClient()
            mock_nel.link = AsyncMock(return_value=_make_nel_response())
            service = ClassifyService(ner_client=mock_ner, nel_client=mock_nel)

            # WHEN classifying with custom options
            options = ClassifyOptions(extract_entities=["occupation"], top_k=3, min_similarity=0.5)
            await service.classify("We need a Head Chef.", options=options)

            # THEN NER is called with the entity type filter
            mock_ner.extract.assert_called_once_with("We need a Head Chef.", ["occupation"])

            # AND NEL is called with the custom top_k and min_similarity
            mock_nel.link.assert_called_once()
            _, kwargs = mock_nel.link.call_args
            assert kwargs["top_k"] == 3
            assert kwargs["min_similarity"] == 0.5

        @pytest.mark.asyncio
        async def test_classify_raises_value_error_for_empty_text(self):
            # GIVEN mock clients
            mock_ner = MockNERClient()
            mock_nel = MockNELClient()
            service = ClassifyService(ner_client=mock_ner, nel_client=mock_nel)

            # WHEN classifying with empty text
            # THEN a ValueError is raised
            with pytest.raises(ValueError):
                await service.classify("")

        @pytest.mark.asyncio
        async def test_classify_propagates_ner_client_error(self):
            # GIVEN the NER client raises an exception
            mock_ner = MockNERClient()
            mock_ner.extract = AsyncMock(side_effect=Exception("NER service unreachable"))
            mock_nel = MockNELClient()
            service = ClassifyService(ner_client=mock_ner, nel_client=mock_nel)

            # WHEN classifying
            # THEN the exception propagates
            with pytest.raises(Exception, match="NER service unreachable"):
                await service.classify("some text")

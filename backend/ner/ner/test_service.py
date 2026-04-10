"""Tests for NERService."""

from unittest.mock import MagicMock

import pytest

from ner.service import NERService


class TestNERService:
    class TestExtractEntities:
        def test_extract_entities_successful(self):
            # GIVEN a mock model that returns two entities
            mock_model = MagicMock()
            mock_model.model_name = "tabiya/roberta-base-job-ner"
            mock_model.extract.return_value = [
                {"entity_type": "occupation", "surface_form": "Head Chef", "span": {"start": 9, "end": 18}},
                {"entity_type": "skill", "surface_form": "plan menus", "span": {"start": 33, "end": 43}},
            ]
            service = NERService(model=mock_model)

            # WHEN extracting entities from a job description
            result = service.extract_entities("We need a Head Chef who can plan menus.")

            # THEN two entities are returned
            assert len(result.entities) == 2
            assert result.metadata.entity_count == 2
            assert result.metadata.model_name == "tabiya/roberta-base-job-ner"
            assert result.metadata.processing_time_ms >= 0

            # AND the model was called with the text
            mock_model.extract.assert_called_once_with("We need a Head Chef who can plan menus.")

        def test_extract_entities_with_entity_type_filter(self):
            # GIVEN a mock model that returns two entities of different types
            mock_model = MagicMock()
            mock_model.model_name = "tabiya/roberta-base-job-ner"
            mock_model.extract.return_value = [
                {"entity_type": "occupation", "surface_form": "Head Chef", "span": {"start": 9, "end": 18}},
                {"entity_type": "skill", "surface_form": "plan menus", "span": {"start": 33, "end": 43}},
            ]
            service = NERService(model=mock_model)

            # WHEN extracting entities filtered to occupation only
            result = service.extract_entities("We need a Head Chef who can plan menus.", entity_types=["occupation"])

            # THEN only occupation entities are returned
            assert len(result.entities) == 1
            assert result.entities[0].entity_type == "occupation"
            assert result.metadata.entity_count == 1

        def test_extract_entities_raises_value_error_for_empty_text(self):
            # GIVEN a service with a loaded model
            mock_model = MagicMock()
            service = NERService(model=mock_model)

            # WHEN extracting entities from an empty string
            # THEN a ValueError is raised
            with pytest.raises(ValueError, match="text"):
                service.extract_entities("")

            # AND the model was never called
            mock_model.extract.assert_not_called()

        def test_extract_entities_raises_runtime_error_when_model_not_loaded(self):
            # GIVEN a service with no model loaded
            service = NERService(model=None)

            # WHEN extracting entities
            # THEN a RuntimeError is raised
            with pytest.raises(RuntimeError, match="not loaded"):
                service.extract_entities("some text")

        def test_extract_entities_returns_empty_list_when_no_entities_found(self):
            # GIVEN a model that returns no entities
            mock_model = MagicMock()
            mock_model.model_name = "tabiya/roberta-base-job-ner"
            mock_model.extract.return_value = []
            service = NERService(model=mock_model)

            # WHEN extracting entities from text
            result = service.extract_entities("Random text with no job entities.")

            # THEN an empty entity list is returned
            assert result.entities == []
            assert result.metadata.entity_count == 0

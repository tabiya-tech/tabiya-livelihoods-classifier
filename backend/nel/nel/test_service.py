"""Tests for NELService."""

from unittest.mock import MagicMock

import pytest

from nel.models import NELOptions
from nel.service import NELService


class TestNELService:
    class TestLinkEntities:
        def test_link_entities_successful(self):
            # GIVEN a mock linker that returns one linked entity
            mock_linker = MagicMock()
            mock_linker.similarity_model_name = "all-MiniLM-L6-v2"
            mock_linker.link.return_value = [
                {
                    "input_text": "Head Chef",
                    "entity_type": "occupation",
                    "matches": [
                        {"similarity_score": 0.91, "taxonomy": "esco", "label": "chef", "code": "3434.1", "uri": None, "eqf_level": None}
                    ],
                }
            ]
            service = NELService(linker=mock_linker)

            # WHEN linking a list of entities
            result = service.link_entities([{"text": "Head Chef", "entity_type": "occupation"}])

            # THEN one linked entity is returned
            assert len(result.linked_entities) == 1
            assert result.linked_entities[0].input_text == "Head Chef"
            assert result.metadata.linker_model == "all-MiniLM-L6-v2"
            assert result.metadata.taxonomy == "esco"

            # AND the linker was called with correct defaults
            mock_linker.link.assert_called_once_with(
                [{"text": "Head Chef", "entity_type": "occupation"}],
                top_k=5,
                min_similarity=0.0,
            )

        def test_link_entities_respects_options(self):
            # GIVEN a mock linker
            mock_linker = MagicMock()
            mock_linker.similarity_model_name = "all-MiniLM-L6-v2"
            mock_linker.link.return_value = []
            service = NELService(linker=mock_linker, max_top_k=50)

            # WHEN linking with custom options
            service.link_entities(
                [{"text": "chef", "entity_type": "occupation"}],
                options=NELOptions(top_k=10, min_similarity=0.7),
            )

            # THEN the linker is called with those options
            mock_linker.link.assert_called_once_with(
                [{"text": "chef", "entity_type": "occupation"}],
                top_k=10,
                min_similarity=0.7,
            )

        def test_link_entities_caps_top_k_at_max(self):
            # GIVEN a service with max_top_k of 10
            mock_linker = MagicMock()
            mock_linker.similarity_model_name = "all-MiniLM-L6-v2"
            mock_linker.link.return_value = []
            service = NELService(linker=mock_linker, max_top_k=10)

            # WHEN linking with top_k exceeding the max
            service.link_entities(
                [{"text": "chef", "entity_type": "occupation"}],
                options=NELOptions(top_k=50),
            )

            # THEN top_k is capped at max_top_k
            mock_linker.link.assert_called_once_with(
                [{"text": "chef", "entity_type": "occupation"}],
                top_k=10,
                min_similarity=0.0,
            )

        def test_link_entities_raises_runtime_error_when_linker_not_loaded(self):
            # GIVEN a service with no linker loaded
            service = NELService(linker=None)

            # WHEN linking entities
            # THEN a RuntimeError is raised
            with pytest.raises(RuntimeError, match="not loaded"):
                service.link_entities([{"text": "chef", "entity_type": "occupation"}])

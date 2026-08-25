"""Tests for per-language model selection in the NER service."""

from unittest.mock import MagicMock

import pytest

from ner.service import NERService


def _mock_model(name: str):
    model = MagicMock()
    model.model_name = name
    model.extract.return_value = []
    return model


class TestPerLanguageModelSelection:
    def test_language_selects_its_own_model(self):
        # GIVEN one model per language behind a provider
        models = {
            "en": _mock_model("tabiya/roberta-base-job-ner"),
            "es": _mock_model("some-org/spanish-job-ner"),
        }
        service = NERService(model_provider=models.__getitem__)

        # WHEN extracting from Spanish text
        result = service.extract_entities("Se busca cocinero con experiencia.", language="es")

        # THEN the Spanish model ran and the English one did not
        models["es"].extract.assert_called_once()
        models["en"].extract.assert_not_called()
        assert result.metadata.model_name == "some-org/spanish-job-ner"
        assert result.metadata.language == "es"

    def test_locale_spelling_is_accepted(self):
        # GIVEN a caller sending a taxonomy locale
        models = {"es": _mock_model("some-org/spanish-job-ner")}
        service = NERService(model_provider=models.__getitem__)

        # WHEN extracting with 'AR-es'
        result = service.extract_entities("Se busca cocinero.", language="AR-es")

        # THEN it resolves to Spanish
        assert result.metadata.language == "es"

    def test_default_language_env_is_used_when_request_omits_it(self, monkeypatch):
        # GIVEN a deployment defaulting to Spanish
        monkeypatch.setenv("TARGET_LANGUAGE", "es")
        models = {"es": _mock_model("some-org/spanish-job-ner")}
        service = NERService(model_provider=models.__getitem__)

        # WHEN extracting without a language
        result = service.extract_entities("Se busca cocinero.")

        # THEN the deployment default applies
        assert result.metadata.language == "es"

    def test_missing_model_for_language_raises_runtime_error(self):
        # GIVEN a provider with no model for the language
        service = NERService(model_provider=lambda _: None)

        # WHEN extracting
        # THEN a RuntimeError naming the language is raised (the route turns it into a 503)
        with pytest.raises(RuntimeError, match="es"):
            service.extract_entities("Se busca cocinero.", language="es")


class TestLanguageSpecificModelFlag:
    def test_spanish_without_its_own_checkpoint_is_flagged(self):
        # GIVEN no NER_MODEL_ES override, so Spanish is served by the English checkpoint
        service = NERService(model_provider=lambda _: _mock_model("tabiya/roberta-base-job-ner"))

        # WHEN extracting Spanish text
        result = service.extract_entities("Se busca cocinero.", language="es")

        # THEN the response says the model is not specific to this language, so the
        # degraded extraction is visible to whoever stores the classification.
        assert result.metadata.model_is_language_specific is False

    def test_english_is_language_specific(self):
        # GIVEN the English model
        service = NERService(model_provider=lambda _: _mock_model("tabiya/roberta-base-job-ner"))

        # WHEN extracting English text
        result = service.extract_entities("We need a head chef.", language="en")

        # THEN it is reported as language-specific
        assert result.metadata.model_is_language_specific is True

    def test_override_marks_spanish_as_language_specific(self, monkeypatch):
        # GIVEN a Spanish checkpoint configured via env
        monkeypatch.setenv("NER_MODEL_ES", "some-org/spanish-job-ner")
        from shared import languages

        languages.get_language_config.cache_clear()
        try:
            service = NERService(model_provider=lambda _: _mock_model("some-org/spanish-job-ner"))

            # WHEN extracting Spanish text
            result = service.extract_entities("Se busca cocinero.", language="es")

            # THEN it is no longer flagged as a cross-language fallback
            assert result.metadata.model_is_language_specific is True
        finally:
            languages.get_language_config.cache_clear()


class TestEntityFilteringStillApplies:
    def test_entity_type_filter_applies_per_language(self):
        # GIVEN a Spanish model returning two entity types
        model = _mock_model("some-org/spanish-job-ner")
        model.extract.return_value = [
            {"entity_type": "occupation", "surface_form": "cocinero", "span": {"start": 0, "end": 8}},
            {"entity_type": "skill", "surface_form": "cocinar", "span": {"start": 9, "end": 16}},
        ]
        service = NERService(model_provider=lambda _: model)

        # WHEN only occupations are requested
        result = service.extract_entities("cocinero cocinar", entity_types=["occupation"], language="es")

        # THEN the filter is applied as it is for English
        assert [e.entity_type for e in result.entities] == ["occupation"]
        assert result.metadata.entity_count == 1

"""Tests for language resolution and propagation in the classify orchestration."""

from unittest.mock import AsyncMock

import pytest

from classify.models import ClassifyOptions
from classify.service import ClassifyService, INELClient, INERClient


class MockNERClient(INERClient):
    async def extract(self, text, entity_types=None, language=None):
        raise NotImplementedError()


class MockNELClient(INELClient):
    async def link(self, entities, top_k, min_similarity, language=None):
        raise NotImplementedError()


def _ner_response():
    return {
        "entities": [
            {"entity_type": "occupation", "surface_form": "cocinero", "span": {"start": 0, "end": 8}},
        ],
        "metadata": {"model_name": "some-org/spanish-job-ner", "entity_count": 1, "processing_time_ms": 10.0},
    }


def _nel_response():
    return {
        "linked_entities": [
            {
                "input_text": "cocinero",
                "entity_type": "occupation",
                "matches": [{"similarity_score": 0.9, "taxonomy": "esco", "label": "cocinero"}],
            }
        ],
        "metadata": {
            "linker_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "taxonomy": "esco",
            "processing_time_ms": 12.0,
            "language": "es",
            "taxonomy_locale": "AR-es",
        },
    }


def _service():
    ner, nel = MockNERClient(), MockNELClient()
    ner.extract = AsyncMock(return_value=_ner_response())
    nel.link = AsyncMock(return_value=_nel_response())
    return ClassifyService(ner_client=ner, nel_client=nel), ner, nel


class TestLanguagePropagation:
    @pytest.mark.asyncio
    async def test_both_stages_get_the_same_language(self):
        # GIVEN a classify service
        service, ner, nel = _service()

        # WHEN classifying with a Spanish option
        await service.classify("Se busca cocinero.", ClassifyOptions(language="es"))

        # THEN NER and NEL are both called with that language — a mismatch would extract
        # in one language and link against another taxonomy.
        assert ner.extract.call_args.kwargs["language"] == "es"
        assert nel.link.call_args.kwargs["language"] == "es"

    @pytest.mark.asyncio
    async def test_locale_spelling_is_normalised_before_the_calls(self):
        # GIVEN a caller that sends the taxonomy locale
        service, ner, nel = _service()

        # WHEN classifying with 'AR-es'
        await service.classify("Se busca cocinero.", ClassifyOptions(language="AR-es"))

        # THEN downstream services receive the language code
        assert ner.extract.call_args.kwargs["language"] == "es"
        assert nel.link.call_args.kwargs["language"] == "es"

    @pytest.mark.asyncio
    async def test_metadata_reports_language_and_taxonomy_locale(self):
        # GIVEN a classify service
        service, _, _ = _service()

        # WHEN classifying in Spanish
        result = await service.classify("Se busca cocinero.", ClassifyOptions(language="es"))

        # THEN the stored metadata records what was actually used
        assert result.metadata.language == "es"
        assert result.metadata.taxonomy_locale == "AR-es"


class TestLanguagePrecedence:
    @pytest.mark.asyncio
    async def test_request_options_win_over_the_caller_default(self):
        # GIVEN an API key configured for English
        service, ner, _ = _service()

        # WHEN the request explicitly asks for Spanish
        await service.classify("Se busca cocinero.", ClassifyOptions(language="es"), language="en")

        # THEN the request wins
        assert ner.extract.call_args.kwargs["language"] == "es"

    @pytest.mark.asyncio
    async def test_caller_default_is_used_when_the_request_is_silent(self):
        # GIVEN an API key configured for Spanish (the per-country key pattern)
        service, ner, _ = _service()

        # WHEN the request carries no language
        await service.classify("Se busca cocinero.", ClassifyOptions(), language="es")

        # THEN the key's language applies without every caller having to say so
        assert ner.extract.call_args.kwargs["language"] == "es"

    @pytest.mark.asyncio
    async def test_falls_back_to_english_when_nothing_is_configured(self):
        # GIVEN neither a request language nor a caller default
        service, ner, _ = _service()

        # WHEN classifying
        await service.classify("We need a head chef.")

        # THEN behaviour is unchanged from before multi-language support
        assert ner.extract.call_args.kwargs["language"] == "en"

    @pytest.mark.asyncio
    async def test_unregistered_language_falls_back_instead_of_failing_the_job(self):
        # GIVEN a request naming a language nobody registered
        service, ner, _ = _service()

        # WHEN classifying
        await service.classify("text", ClassifyOptions(language="klingon"))

        # THEN the job is classified in the default language rather than erroring
        assert ner.extract.call_args.kwargs["language"] == "en"


class TestNoEntitiesPath:
    @pytest.mark.asyncio
    async def test_language_metadata_present_when_nothing_is_linkable(self):
        # GIVEN NER finds nothing linkable, so NEL is never called
        ner, nel = MockNERClient(), MockNELClient()
        ner.extract = AsyncMock(
            return_value={"entities": [], "metadata": {"model_name": "m", "entity_count": 0, "processing_time_ms": 1.0}}
        )
        nel.link = AsyncMock()
        service = ClassifyService(ner_client=ner, nel_client=nel)

        # WHEN classifying in Spanish
        result = await service.classify("...", ClassifyOptions(language="es"))

        # THEN the language is still recorded, from the language config rather than NEL
        nel.link.assert_not_called()
        assert result.metadata.language == "es"
        assert result.metadata.taxonomy_locale == "AR-es"

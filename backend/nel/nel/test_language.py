"""Tests for per-language linking in the NEL service and route."""

from http import HTTPStatus
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nel.models import LinkedEntity, NELMetadata, NELResponse
from nel.service import INELService, NELService


def _mock_linker(model_name: str):
    linker = MagicMock()
    linker.similarity_model_name = model_name
    linker.link.return_value = []
    return linker


class TestPerLanguageLinkerSelection:
    def test_language_selects_its_own_linker(self):
        # GIVEN one linker per language behind a provider
        linkers = {
            "en": _mock_linker("all-MiniLM-L6-v2"),
            "es": _mock_linker("paraphrase-multilingual-MiniLM-L12-v2"),
        }
        service = NELService(linker_provider=linkers.__getitem__)

        # WHEN linking Spanish text
        result = service.link_entities([{"text": "cocinero", "entity_type": "occupation"}], language="es")

        # THEN the Spanish linker ran and the English one did not
        linkers["es"].link.assert_called_once()
        linkers["en"].link.assert_not_called()

        # AND the response reports which language and taxonomy locale were used
        assert result.metadata.language == "es"
        assert result.metadata.taxonomy_locale == "AR-es"
        assert result.metadata.linker_model == "paraphrase-multilingual-MiniLM-L12-v2"

    def test_taxonomy_locale_is_reported_for_english(self):
        # GIVEN an English-only service
        service = NELService(linker_provider=lambda _: _mock_linker("all-MiniLM-L6-v2"))

        # WHEN linking without a language
        result = service.link_entities([{"text": "chef", "entity_type": "occupation"}])

        # THEN English is used
        assert result.metadata.language == "en"
        assert result.metadata.taxonomy_locale == "en"

    def test_taxonomy_locale_spelling_is_accepted(self):
        # GIVEN a caller that sends the taxonomy locale rather than a language code
        linkers = {"es": _mock_linker("paraphrase-multilingual-MiniLM-L12-v2")}
        service = NELService(linker_provider=linkers.__getitem__)

        # WHEN linking with 'AR-es'
        result = service.link_entities([{"text": "cocinero", "entity_type": "skill"}], language="AR-es")

        # THEN it resolves to Spanish
        assert result.metadata.language == "es"

    def test_default_language_env_is_used_when_request_omits_it(self, monkeypatch):
        # GIVEN a deployment whose default language is Spanish
        monkeypatch.setenv("TARGET_LANGUAGE", "es")
        linkers = {"es": _mock_linker("paraphrase-multilingual-MiniLM-L12-v2")}
        service = NELService(linker_provider=linkers.__getitem__)

        # WHEN linking without a language on the request
        result = service.link_entities([{"text": "cocinero", "entity_type": "skill"}])

        # THEN the deployment default applies
        assert result.metadata.language == "es"

    def test_missing_linker_for_language_raises_runtime_error(self):
        # GIVEN a provider with no linker for the requested language
        service = NELService(linker_provider=lambda _: None)

        # WHEN linking
        # THEN a RuntimeError naming the language is raised (the route turns it into a 503)
        try:
            service.link_entities([{"text": "cocinero", "entity_type": "skill"}], language="es")
        except RuntimeError as e:
            assert "es" in str(e)
        else:
            raise AssertionError("expected RuntimeError")


class TestLanguageOnTheRoute:
    def test_route_forwards_the_request_language(self, client_with_mocks: tuple[TestClient, INELService]):
        client, mock_service = client_with_mocks
        # GIVEN a service that returns an empty result
        mock_service.link_entities = MagicMock(
            return_value=NELResponse(
                linked_entities=[],
                metadata=NELMetadata(
                    linker_model="paraphrase-multilingual-MiniLM-L12-v2",
                    taxonomy="esco",
                    processing_time_ms=1.0,
                    language="es",
                    taxonomy_locale="AR-es",
                ),
            )
        )

        # WHEN a request carries a taxonomy locale as its language
        response = client.post(
            "/v1/nel",
            json={"entities": [{"text": "cocinero", "entity_type": "occupation"}], "language": "AR-es"},
        )

        # THEN it is normalised to the language code before reaching the service
        assert response.status_code == HTTPStatus.OK
        assert mock_service.link_entities.call_args.kwargs["language"] == "es"

    def test_unknown_language_falls_back_rather_than_erroring(
        self, client_with_mocks: tuple[TestClient, INELService]
    ):
        client, mock_service = client_with_mocks
        # GIVEN a service that accepts any language
        mock_service.link_entities = MagicMock(
            return_value=NELResponse(
                linked_entities=[],
                metadata=NELMetadata(
                    linker_model="all-MiniLM-L6-v2", taxonomy="esco", processing_time_ms=1.0
                ),
            )
        )

        # WHEN a request names a language that is not registered
        response = client.post(
            "/v1/nel",
            json={"entities": [{"text": "chef", "entity_type": "occupation"}], "language": "fr"},
        )

        # THEN the request still succeeds against the default language
        assert response.status_code == HTTPStatus.OK
        assert mock_service.link_entities.call_args.kwargs["language"] == "en"


class TestLinkedEntityShapeIsUnchanged:
    def test_matches_pass_through_untouched(self):
        # GIVEN a linker returning one match
        linker = _mock_linker("paraphrase-multilingual-MiniLM-L12-v2")
        linker.link.return_value = [
            {
                "input_text": "cocinero",
                "entity_type": "occupation",
                "matches": [
                    {
                        "similarity_score": 0.88,
                        "taxonomy": "esco",
                        "label": "cocinero",
                        "code": "5120.1",
                        "uri": "http://data.europa.eu/esco/occupation/abc",
                    }
                ],
            }
        ]
        service = NELService(linker_provider=lambda _: linker)

        # WHEN linking in Spanish
        result = service.link_entities([{"text": "cocinero", "entity_type": "occupation"}], language="es")

        # THEN the linked entity shape downstream stages read is unchanged
        assert isinstance(result.linked_entities[0], LinkedEntity)
        assert result.linked_entities[0].matches[0].label == "cocinero"
        assert result.linked_entities[0].matches[0].uri.endswith("/abc")

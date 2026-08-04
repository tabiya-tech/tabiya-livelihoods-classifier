"""Tests for the language registry."""

import pytest

from shared import languages
from shared.languages import (
    LANGUAGE_REGISTRY,
    default_language,
    enabled_languages,
    get_language_config,
    normalise_language,
)


@pytest.fixture(autouse=True)
def _clear_language_env(monkeypatch):
    """Language resolution reads env; keep tests independent of the developer's shell."""
    for var in ("TARGET_LANGUAGE", "LANGUAGE", "ENABLED_LANGUAGES"):
        monkeypatch.delenv(var, raising=False)
    # get_language_config / _locale_index are lru_cached and the configs read env at
    # import time, so a test that changes a model env var must not leak into the next.
    languages.get_language_config.cache_clear()
    languages._locale_index.cache_clear()
    yield
    languages.get_language_config.cache_clear()
    languages._locale_index.cache_clear()


class TestRegistry:
    def test_every_registered_language_has_a_config(self):
        # GIVEN the registry
        # WHEN each language's config is loaded
        # THEN it exposes the keys every service reads
        for code in LANGUAGE_REGISTRY:
            cfg = get_language_config(code)
            assert cfg["language"] == code
            for key in ("name", "locales", "ner_model", "nel_similarity_model", "nel_files_subdir"):
                assert cfg.get(key), f"{code} config is missing {key}"

    def test_unregistered_language_raises(self):
        # GIVEN a language that is not registered
        # WHEN its config is requested
        # THEN it raises rather than silently serving another language's data
        with pytest.raises(ValueError, match="Unsupported language"):
            get_language_config("fr")

    def test_english_defaults_are_the_pre_language_behaviour(self):
        # GIVEN no language env vars set
        # WHEN the English config is read
        cfg = get_language_config("en")

        # THEN it still points at the models the services hardcoded before
        assert cfg["ner_model"] == "tabiya/roberta-base-job-ner"
        assert cfg["nel_similarity_model"] == "all-MiniLM-L6-v2"


class TestNormaliseLanguage:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("en", "en"),
            ("EN", "en"),
            ("english", "en"),
            ("en-KE", "en"),
            ("es", "es"),
            ("ES", "es"),
            ("spanish", "es"),
            ("AR-es", "es"),  # taxonomy export locale
            ("es_AR", "es"),
            ("es_AR.UTF-8", "es"),
            ("es-419", "es"),
            ("argentina", "es"),  # TARGET_COUNTRY spelling
        ],
    )
    def test_resolves_locale_spellings(self, value, expected):
        # GIVEN a locale spelling a caller might send
        # WHEN it is normalised
        # THEN it resolves to the registered language code
        assert normalise_language(value) == expected

    def test_none_uses_the_default(self):
        # GIVEN no language on the request
        # WHEN normalised
        # THEN the service default is used
        assert normalise_language(None) == "en"

    def test_unknown_language_falls_back_instead_of_raising(self):
        # GIVEN an unregistered language on a request
        # WHEN normalised
        # THEN it falls back rather than failing the job
        assert normalise_language("klingon") == "en"
        assert normalise_language("klingon", fallback="es") == "es"


class TestDefaultLanguage:
    def test_defaults_to_english(self):
        # GIVEN no env override
        # THEN English is the default, so existing deployments are unaffected
        assert default_language() == "en"

    def test_target_language_wins_over_language(self, monkeypatch):
        # GIVEN both variables set
        monkeypatch.setenv("TARGET_LANGUAGE", "es")
        monkeypatch.setenv("LANGUAGE", "en")

        # THEN TARGET_LANGUAGE decides
        assert default_language() == "es"

    def test_locale_is_accepted(self, monkeypatch):
        # GIVEN a taxonomy locale rather than a bare code
        monkeypatch.setenv("TARGET_LANGUAGE", "AR-es")

        # THEN it resolves
        assert default_language() == "es"

    def test_unregistered_value_falls_back_to_english(self, monkeypatch):
        # GIVEN a language nobody has registered
        monkeypatch.setenv("TARGET_LANGUAGE", "fr")

        # THEN English is used rather than crashing at startup
        assert default_language() == "en"


class TestEnabledLanguages:
    def test_defaults_to_the_default_language_only(self, monkeypatch):
        # GIVEN no ENABLED_LANGUAGES
        monkeypatch.setenv("TARGET_LANGUAGE", "es")

        # THEN only that language is warmed up (each extra one costs a model load)
        assert enabled_languages() == ("es",)

    def test_explicit_list(self, monkeypatch):
        # GIVEN a list, with spacing and a locale spelling
        monkeypatch.setenv("ENABLED_LANGUAGES", " en , AR-es ")

        # THEN both resolve, in the order given
        assert enabled_languages() == ("en", "es")

    def test_all_means_every_registered_language(self, monkeypatch):
        # GIVEN the 'all' shorthand
        monkeypatch.setenv("ENABLED_LANGUAGES", "all")

        # THEN every registered language is warmed up
        assert enabled_languages() == LANGUAGE_REGISTRY

    def test_unregistered_entries_are_ignored(self, monkeypatch):
        # GIVEN a list containing an unregistered language
        monkeypatch.setenv("ENABLED_LANGUAGES", "en,fr")

        # THEN it is dropped and the valid ones are kept
        assert enabled_languages() == ("en",)

    def test_all_unregistered_falls_back_to_default(self, monkeypatch):
        # GIVEN a list with nothing usable in it
        monkeypatch.setenv("ENABLED_LANGUAGES", "fr,de")

        # THEN the service still warms up its default language
        assert enabled_languages() == ("en",)


class TestPerLanguageEnvOverrides:
    def test_language_suffixed_variable_wins(self, monkeypatch):
        # GIVEN a per-language model override
        monkeypatch.setenv("NER_MODEL_ES", "some-org/spanish-job-ner")
        languages.get_language_config.cache_clear()

        # WHEN the Spanish config is read
        cfg = get_language_config("es")

        # THEN the override is used, and the model counts as language-specific
        assert cfg["ner_model"] == "some-org/spanish-job-ner"
        assert cfg["ner_model_is_language_specific"] is True

    def test_spanish_ner_falls_back_to_english_model_and_says_so(self):
        # GIVEN no NER_MODEL_ES
        # WHEN the Spanish config is read
        cfg = get_language_config("es")

        # THEN it is served by the English checkpoint, flagged as not language-specific
        # so consumers can see that extraction quality is degraded.
        assert cfg["ner_model"] == "tabiya/roberta-base-job-ner"
        assert cfg["ner_model_is_language_specific"] is False

    def test_bare_variable_applies_to_every_language(self, monkeypatch):
        # GIVEN the pre-language variable name (what current deployments set)
        monkeypatch.setenv("LINKER_MODEL", "custom/model")
        languages.get_language_config.cache_clear()

        # THEN it is honoured, so existing config keeps working
        assert get_language_config("en")["nel_similarity_model"] == "custom/model"

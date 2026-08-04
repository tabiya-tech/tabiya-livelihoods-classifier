"""NER service: interface and implementation."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from ner.models import NERMetadata, NERResponse
from shared.languages import default_language, get_language_config, normalise_language


class INERService(ABC):
    @abstractmethod
    def extract_entities(
        self,
        text: str,
        entity_types: Optional[list[str]] = None,
        language: Optional[str] = None,
    ) -> NERResponse:
        """
        Extract entities from job-related text.
        :param text: The text to extract entities from.
        :param entity_types: Optional filter to only return specific entity types.
        :param language: Language of the text; None uses the service default.
        :raises ValueError: if text is empty.
        :raises RuntimeError: if the model for that language is not loaded.
        """
        raise NotImplementedError()


class NERService(INERService):
    """Extracts entities with the model that matches the request's language.

    Pass ``model_provider`` (``language -> model``) to serve several languages from one
    service instance, or ``model`` to pin every request to a single loaded model.
    """

    def __init__(
        self,
        model: Any = None,
        model_provider: Optional[Callable[[str], Any]] = None,
    ):
        self._model = model
        self._model_provider = model_provider
        self._logger = logging.getLogger(self.__class__.__name__)

    def _resolve_model(self, language: str) -> Any:
        if self._model_provider is not None:
            return self._model_provider(language)
        return self._model

    def extract_entities(
        self,
        text: str,
        entity_types: Optional[list[str]] = None,
        language: Optional[str] = None,
    ) -> NERResponse:
        if not text:
            raise ValueError("Field 'text' is required and cannot be empty")

        lang = normalise_language(language) if language else default_language()
        model = self._resolve_model(lang)
        if model is None:
            raise RuntimeError(f"NER model is not loaded (language={lang})")

        start = time.time()
        entities = model.extract(text)
        processing_time = round((time.time() - start) * 1000, 1)

        if entity_types:
            allowed = {t.lower() for t in entity_types}
            entities = [e for e in entities if e["entity_type"] in allowed]

        self._logger.info(
            "NER done: %d entities in %.1fms (language=%s)", len(entities), processing_time, lang
        )

        cfg = get_language_config(lang)
        return NERResponse(
            entities=entities,
            metadata=NERMetadata(
                model_name=model.model_name,
                entity_count=len(entities),
                processing_time_ms=processing_time,
                language=lang,
                # False when this language has no checkpoint of its own and is being
                # served by another language's model — extraction quality will be poor,
                # and consumers should be able to see that in the stored output.
                model_is_language_specific=bool(
                    cfg.get("ner_model_is_language_specific", True)
                ),
            ),
        )

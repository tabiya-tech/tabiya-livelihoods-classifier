"""NEL service: interface and implementation."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from nel.models import NELOptions, NELResponse, NELMetadata
from shared.languages import default_language, get_language_config, normalise_language


class INELService(ABC):
    @abstractmethod
    def link_entities(
        self,
        entities: list[dict],
        options: Optional[NELOptions] = None,
        language: Optional[str] = None,
    ) -> NELResponse:
        """
        Link entities to ESCO taxonomy entries.
        :param entities: List of dicts with 'text' and 'entity_type' keys.
        :param options: Optional top_k and min_similarity settings.
        :param language: Language of the entity text; None uses the service default.
        :raises RuntimeError: if the linker for that language is not loaded.
        """
        raise NotImplementedError()


class NELService(INELService):
    """Links entities with the linker that matches the request's language.

    Pass ``linker_provider`` (``language -> linker``) to serve several languages from one
    service instance, or ``linker`` to pin every request to a single loaded linker.
    """

    def __init__(
        self,
        linker: Any = None,
        max_top_k: int = 50,
        linker_provider: Optional[Callable[[str], Any]] = None,
    ):
        self._linker = linker
        self._linker_provider = linker_provider
        self._max_top_k = max_top_k
        self._logger = logging.getLogger(self.__class__.__name__)

    def _resolve_linker(self, language: str) -> Any:
        if self._linker_provider is not None:
            return self._linker_provider(language)
        return self._linker

    def link_entities(
        self,
        entities: list[dict],
        options: Optional[NELOptions] = None,
        language: Optional[str] = None,
    ) -> NELResponse:
        lang = normalise_language(language) if language else default_language()
        linker = self._resolve_linker(lang)
        if linker is None:
            raise RuntimeError(f"NEL linker is not loaded (language={lang})")

        opts = options or NELOptions()
        top_k = min(opts.top_k, self._max_top_k)
        min_similarity = opts.min_similarity

        self._logger.info(
            "NEL request: %d entities, top_k=%d, language=%s", len(entities), top_k, lang
        )
        start = time.time()

        results = linker.link(entities, top_k=top_k, min_similarity=min_similarity)

        processing_time = round((time.time() - start) * 1000, 1)
        self._logger.info("NEL done: %d linked in %.1fms", len(results), processing_time)

        return NELResponse(
            linked_entities=results,
            metadata=NELMetadata(
                linker_model=linker.similarity_model_name,
                taxonomy="esco",
                processing_time_ms=processing_time,
                language=lang,
                taxonomy_locale=get_language_config(lang).get("taxonomy_locale", lang),
            ),
        )

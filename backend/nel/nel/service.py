"""NEL service: interface and implementation."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from nel.models import NELOptions, NELResponse, NELMetadata


class INELService(ABC):
    @abstractmethod
    def link_entities(self, entities: list[dict], options: Optional[NELOptions] = None) -> NELResponse:
        """
        Link entities to ESCO taxonomy entries.
        :param entities: List of dicts with 'text' and 'entity_type' keys.
        :param options: Optional top_k and min_similarity settings.
        :raises RuntimeError: if the linker is not loaded.
        """
        raise NotImplementedError()


class NELService(INELService):
    def __init__(self, linker, max_top_k: int = 50):
        self._linker = linker
        self._max_top_k = max_top_k
        self._logger = logging.getLogger(self.__class__.__name__)

    def link_entities(self, entities: list[dict], options: Optional[NELOptions] = None) -> NELResponse:
        if self._linker is None:
            raise RuntimeError("NEL linker is not loaded")

        opts = options or NELOptions()
        top_k = min(opts.top_k, self._max_top_k)
        min_similarity = opts.min_similarity

        self._logger.info("NEL request: %d entities, top_k=%d", len(entities), top_k)
        start = time.time()

        results = self._linker.link(entities, top_k=top_k, min_similarity=min_similarity)

        processing_time = round((time.time() - start) * 1000, 1)
        self._logger.info("NEL done: %d linked in %.1fms", len(results), processing_time)

        return NELResponse(
            linked_entities=results,
            metadata=NELMetadata(
                linker_model=self._linker.similarity_model_name,
                taxonomy="esco",
                processing_time_ms=processing_time,
            ),
        )

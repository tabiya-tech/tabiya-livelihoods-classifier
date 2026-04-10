"""NER service: interface and implementation."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from ner.models import NERMetadata, NERResponse


class INERService(ABC):
    @abstractmethod
    def extract_entities(self, text: str, entity_types: Optional[list[str]] = None) -> NERResponse:
        """
        Extract entities from job-related text.
        :param text: The text to extract entities from.
        :param entity_types: Optional filter to only return specific entity types.
        :raises ValueError: if text is empty.
        :raises RuntimeError: if the model is not loaded.
        """
        raise NotImplementedError()


class NERService(INERService):
    def __init__(self, model):
        self._model = model
        self._logger = logging.getLogger(self.__class__.__name__)

    def extract_entities(self, text: str, entity_types: Optional[list[str]] = None) -> NERResponse:
        if not text:
            raise ValueError("Field 'text' is required and cannot be empty")

        if self._model is None:
            raise RuntimeError("NER model is not loaded")

        start = time.time()
        entities = self._model.extract(text)
        processing_time = round((time.time() - start) * 1000, 1)

        if entity_types:
            allowed = {t.lower() for t in entity_types}
            entities = [e for e in entities if e["entity_type"] in allowed]

        self._logger.info("NER done: %d entities in %.1fms", len(entities), processing_time)

        return NERResponse(
            entities=entities,
            metadata=NERMetadata(
                model_name=self._model.model_name,
                entity_count=len(entities),
                processing_time_ms=processing_time,
            ),
        )

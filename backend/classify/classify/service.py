"""Classify service: interface, NER/NEL client interfaces, and implementation."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from classify.models import ClassifyMetadata, ClassifyOptions, ClassifyResponse, Classification
from shared.job_text import compute_hash
from shared.languages import get_language_config, normalise_language
from classify.config import CLASSIFIER_VERSION

log = logging.getLogger("classify-service")


class INERClient(ABC):
    """HTTP client interface for the NER service."""

    @abstractmethod
    async def extract(
        self,
        text: str,
        entity_types: Optional[list[str]] = None,
        language: Optional[str] = None,
    ) -> dict:
        """
        Call the NER service and return the raw response dict.
        :raises httpx.HTTPStatusError: on non-2xx responses.
        :raises httpx.RequestError: if the NER service is unreachable.
        """
        raise NotImplementedError()


class INELClient(ABC):
    """HTTP client interface for the NEL service."""

    @abstractmethod
    async def link(
        self,
        entities: list[dict],
        top_k: int,
        min_similarity: float,
        language: Optional[str] = None,
    ) -> dict:
        """
        Call the NEL service and return the raw response dict.
        :raises httpx.HTTPStatusError: on non-2xx responses.
        :raises httpx.RequestError: if the NEL service is unreachable.
        """
        raise NotImplementedError()


class IClassifyService(ABC):
    @abstractmethod
    async def classify(
        self,
        input_text: str,
        options: Optional[ClassifyOptions] = None,
        language: Optional[str] = None,
    ) -> ClassifyResponse:
        """
        Orchestrate NER → NEL and return a merged classification result.
        :param input_text: The job text to classify.
        :param options: Optional classification settings.
        :param language: Language of the text. ``options.language`` wins when both are
            set; None on both uses the service default.
        :raises ValueError: if input_text is empty.
        """
        raise NotImplementedError()


class ClassifyService(IClassifyService):
    def __init__(self, ner_client: INERClient, nel_client: INELClient):
        self._ner = ner_client
        self._nel = nel_client
        self._logger = logging.getLogger(self.__class__.__name__)

    async def classify(
        self,
        input_text: str,
        options: Optional[ClassifyOptions] = None,
        language: Optional[str] = None,
    ) -> ClassifyResponse:
        if not input_text:
            raise ValueError("input_text cannot be empty")

        opts = options or ClassifyOptions()
        start = time.time()

        # Precedence: request options > caller default (API key config) > service default.
        # Resolved here rather than in NER/NEL so both stages see the same language.
        lang = normalise_language(opts.language or language)

        entity_type_filter = [e.value for e in opts.extract_entities] if opts.extract_entities else None
        ner_data = await self._ner.extract(input_text, entity_type_filter, language=lang)
        ner_entities = ner_data.get("entities", [])

        linkable_types = {"occupation", "skill", "qualification"}
        nel_input = [
            {"text": e["surface_form"], "entity_type": e["entity_type"]}
            for e in ner_entities
            if e["entity_type"] in linkable_types
        ]

        linked_map: dict = {}
        nel_metadata: dict = {}
        if nel_input:
            nel_data = await self._nel.link(
                nel_input,
                top_k=opts.top_k,
                min_similarity=opts.min_similarity,
                language=lang,
            )
            nel_metadata = nel_data.get("metadata", {})
            for item in nel_data.get("linked_entities", []):
                key = (item["input_text"], item["entity_type"])
                linked_map[key] = item["matches"]

        merged_entities = []
        entity_counts: dict[str, int] = {}
        for entity in ner_entities:
            etype = entity["entity_type"]
            entity_counts[etype] = entity_counts.get(etype, 0) + 1
            merged = {
                "entity_type": etype,
                "surface_form": entity["surface_form"],
                "span": entity["span"],
            }
            key = (entity["surface_form"], etype)
            if key in linked_map:
                merged["linked_entities"] = linked_map[key]
            merged_entities.append(merged)

        processing_time = round((time.time() - start) * 1000, 1)
        self._logger.info(
            "Classify done: %d entities in %.1fms (language=%s)",
            len(merged_entities),
            processing_time,
            lang,
        )

        return ClassifyResponse(
            classification=Classification(
                entities=merged_entities,
                entity_counts=entity_counts,
            ),
            metadata=ClassifyMetadata(
                classifier_version=CLASSIFIER_VERSION,
                model_name=ner_data.get("metadata", {}).get("model_name", "unknown"),
                linker_model=nel_metadata.get("linker_model", "unknown"),
                processing_time_ms=processing_time,
                input_text_hash=compute_hash(input_text),
                language=lang,
                taxonomy_locale=nel_metadata.get("taxonomy_locale")
                or get_language_config(lang).get("taxonomy_locale", lang),
            ),
        )

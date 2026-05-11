"""Classify service: interface, NER/NEL client interfaces, and implementation."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from classify.models import ClassifyMetadata, ClassifyOptions, ClassifyResponse, Classification
from shared.job_text import compute_hash
from classify.config import CLASSIFIER_VERSION

log = logging.getLogger("classify-service")


class INERClient(ABC):
    """HTTP client interface for the NER service."""

    @abstractmethod
    async def extract(self, text: str, entity_types: Optional[list[str]] = None) -> dict:
        """
        Call the NER service and return the raw response dict.
        :raises httpx.HTTPStatusError: on non-2xx responses.
        :raises httpx.RequestError: if the NER service is unreachable.
        """
        raise NotImplementedError()


class INELClient(ABC):
    """HTTP client interface for the NEL service."""

    @abstractmethod
    async def link(self, entities: list[dict], top_k: int, min_similarity: float) -> dict:
        """
        Call the NEL service and return the raw response dict.
        :raises httpx.HTTPStatusError: on non-2xx responses.
        :raises httpx.RequestError: if the NEL service is unreachable.
        """
        raise NotImplementedError()


class IClassifyService(ABC):
    @abstractmethod
    async def classify(self, input_text: str, options: Optional[ClassifyOptions] = None) -> ClassifyResponse:
        """
        Orchestrate NER → NEL and return a merged classification result.
        :param input_text: The job text to classify.
        :param options: Optional classification settings.
        :raises ValueError: if input_text is empty.
        """
        raise NotImplementedError()


class ClassifyService(IClassifyService):
    def __init__(self, ner_client: INERClient, nel_client: INELClient):
        self._ner = ner_client
        self._nel = nel_client
        self._logger = logging.getLogger(self.__class__.__name__)

    async def classify(self, input_text: str, options: Optional[ClassifyOptions] = None) -> ClassifyResponse:
        if not input_text:
            raise ValueError("input_text cannot be empty")

        opts = options or ClassifyOptions()
        start = time.time()

        entity_type_filter = [e.value for e in opts.extract_entities] if opts.extract_entities else None
        ner_data = await self._ner.extract(input_text, entity_type_filter)
        ner_entities = ner_data.get("entities", [])

        linkable_types = {"occupation", "skill", "qualification"}
        nel_input: list[dict] = []
        # Parallel to nel_input: index into ner_entities so each occurrence gets its own NEL
        # matches (duplicate surface_form + type no longer collapse into one map entry).
        nel_source_indices: list[int] = []
        for i, e in enumerate(ner_entities):
            if e["entity_type"] in linkable_types:
                nel_input.append({"text": e["surface_form"], "entity_type": e["entity_type"]})
                nel_source_indices.append(i)

        linked_by_ner_index: dict[int, list] = {}
        nel_metadata: dict = {}
        if nel_input:
            nel_data = await self._nel.link(nel_input, top_k=opts.top_k, min_similarity=opts.min_similarity)
            nel_metadata = nel_data.get("metadata", {})
            linked_results = nel_data.get("linked_entities", [])
            n_expected = len(nel_source_indices)
            n_got = len(linked_results)
            if n_got != n_expected:
                self._logger.warning(
                    "NEL linked_entities length mismatch: got %d, expected %d (using positional alignment up to min).",
                    n_got,
                    n_expected,
                )
            for j, item in enumerate(linked_results):
                if j >= len(nel_source_indices):
                    break
                ner_i = nel_source_indices[j]
                linked_by_ner_index[ner_i] = item.get("matches", [])

        merged_entities = []
        entity_counts: dict[str, int] = {}
        for i, entity in enumerate(ner_entities):
            etype = entity["entity_type"]
            entity_counts[etype] = entity_counts.get(etype, 0) + 1
            merged = {
                "entity_type": etype,
                "surface_form": entity["surface_form"],
                "span": entity["span"],
            }
            if i in linked_by_ner_index:
                merged["linked_entities"] = linked_by_ner_index[i]
            merged_entities.append(merged)

        processing_time = round((time.time() - start) * 1000, 1)
        self._logger.info("Classify done: %d entities in %.1fms", len(merged_entities), processing_time)

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
            ),
        )

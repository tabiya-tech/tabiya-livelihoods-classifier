"""Entity linking repository — wraps vector search into typed match results."""

from abc import ABC, abstractmethod

from nel.app.embeddings_cache.repository.repository import IEmbeddingsCacheRepository
from nel.app.embeddings_cache.types import EmbeddingDocument
from nel.app.linking.service.types import (
    OccupationEntity,
    OccupationMatch,
    QualificationEntity,
    QualificationMatch,
    SkillEntity,
    SkillMatch,
)

def _doc_to_match(doc: EmbeddingDocument, score: float) -> OccupationMatch | SkillMatch | QualificationMatch:
    base = {
        "uuid": doc.entity_uuid,
        "origin_uuid": doc.origin_uuid,
        "uuid_history": doc.uuid_history,
        "preferred_label": doc.preferred_label,
        "origin_uri": doc.origin_uri,
        "alt_labels": doc.alt_labels,
        "description": doc.description,
    }
    if doc.entity_type == "occupation":
        return OccupationMatch(
            similarity_score=score,
            entity=OccupationEntity(**base, esco_code=doc.esco_code),
        )
    if doc.entity_type == "skill":
        return SkillMatch(
            similarity_score=score,
            entity=SkillEntity(**base, skill_type=doc.skill_type, reuse_level=doc.reuse_level),
        )
    # qualification
    return QualificationMatch(
        similarity_score=score,
        entity=QualificationEntity(**base, eqf_level=doc.eqf_level, country=doc.country),
    )


class IEntityLinkingRepository(ABC):
    @abstractmethod
    async def find_matches(
        self,
        entity_type: str,
        query_embedding: list[float],
        taxonomy_model_id: str,
        nel_model_id: str,
        top_k: int,
        min_similarity: float,
    ) -> list[OccupationMatch | SkillMatch | QualificationMatch]: ...


class EntityLinkingRepository(IEntityLinkingRepository):
    _INDEX_NAME = "nel_embedding_index_384"

    def __init__(self, cache_repository: IEmbeddingsCacheRepository):
        self._cache_repo = cache_repository

    async def find_matches(
        self,
        entity_type: str,
        query_embedding: list[float],
        taxonomy_model_id: str,
        nel_model_id: str,
        top_k: int,
        min_similarity: float,
    ) -> list[OccupationMatch | SkillMatch | QualificationMatch]:
        results = await self._cache_repo.vector_search(
            entity_type=entity_type,
            query_embedding=query_embedding,
            taxonomy_model_id=taxonomy_model_id,
            nel_model_id=nel_model_id,
            top_k=top_k,
            min_similarity=min_similarity,
            index_name=self._INDEX_NAME,
        )
        return [_doc_to_match(doc, score) for doc, score in results]

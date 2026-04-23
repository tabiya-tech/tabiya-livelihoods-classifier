"""Entity linking routes."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException

from nel.app.embedding.service.service import get_embedding_service
from nel.app.embeddings_cache.repository.repository import EmbeddingsCacheRepository
from nel.app.linking.repository.repository import EntityLinkingRepository
from nel.app.linking.routes._types import NELRequest
from nel.app.linking.service.errors import EmbeddingsCacheNotReadyError
from nel.app.linking.service.service import INELService, NELService
from nel.app.linking.service.types import NELOptions, NELResponse
from nel.app.server_dependencies.db_dependencies import ClassifierDBProvider

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/nel", tags=["nel"])


async def _get_service(x_nel_model_id: Annotated[str | None, Header()] = None) -> INELService:
    from nel.config import DEFAULT_NEL_MODEL_ID
    nel_model_id = x_nel_model_id or DEFAULT_NEL_MODEL_ID
    app_db = await ClassifierDBProvider.get_application_db()
    taxonomy_db = await ClassifierDBProvider.get_taxonomy_db()
    cache_repo = EmbeddingsCacheRepository(app_db=app_db, taxonomy_db=taxonomy_db)
    linking_repo = EntityLinkingRepository(cache_repo)
    embedding_svc = await get_embedding_service(nel_model_id)
    return NELService(
        linking_repository=linking_repo,
        cache_repository=cache_repo,
        embedding_service=embedding_svc,
    )


@router.post("", response_model=NELResponse)
async def link_entities(
    request: NELRequest,
    x_taxonomy_model_id: Annotated[str | None, Header()] = None,
    x_nel_model_id: Annotated[str | None, Header()] = None,
    svc: INELService = Depends(_get_service),
):
    from nel.config import DEFAULT_TAXONOMY_MODEL_ID, DEFAULT_NEL_MODEL_ID

    taxonomy_model_id = x_taxonomy_model_id or DEFAULT_TAXONOMY_MODEL_ID
    nel_model_id = x_nel_model_id or DEFAULT_NEL_MODEL_ID

    if not taxonomy_model_id:
        raise HTTPException(
            status_code=400,
            detail="taxonomy_model_id required — pass X-Taxonomy-Model-Id header or set DEFAULT_TAXONOMY_MODEL_ID",
        )

    try:
        return await svc.link(
            entities=[(e.text, e.entity_type) for e in request.entities],
            taxonomy_model_id=taxonomy_model_id,
            nel_model_id=nel_model_id,
            options=NELOptions(top_k=request.top_k, min_similarity=request.min_similarity),
        )
    except EmbeddingsCacheNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

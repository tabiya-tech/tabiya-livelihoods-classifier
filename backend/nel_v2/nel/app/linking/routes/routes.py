"""Entity linking routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from nel.app.embedding.service.service import get_embedding_service
from nel.app.embeddings_cache.repository.repository import EmbeddingsCacheRepository
from nel.app.linking.repository.repository import EntityLinkingRepository
from nel.app.linking.routes._types import NELRequest
from nel.app.linking.service.errors import EmbeddingsCacheNotReadyError
from nel.app.linking.service.service import INELService, NELService
from nel.app.linking.service.types import NELOptions, NELResponse
from nel.app.server_dependencies.db_dependencies import ClassifierDBProvider
from nel.app.user_config.repository.repository import UserConfigRepository
from nel.app.user_config.routes.auth import get_firebase_uid
from nel.app.user_config.service.service import UserConfigService
from nel.app.user_config.service.types import UserConfig

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/nel", tags=["nel"])


async def _get_user_config(uid: str = Depends(get_firebase_uid)) -> UserConfig:
    app_db = await ClassifierDBProvider.get_application_db()
    return await UserConfigService(UserConfigRepository(app_db)).get(uid)


async def _get_service(user_config: UserConfig = Depends(_get_user_config)) -> INELService:
    from nel.config import DEFAULT_NEL_MODEL_ID
    nel_model_id = user_config.nel_model_id or DEFAULT_NEL_MODEL_ID

    app_db = await ClassifierDBProvider.get_application_db()
    taxonomy_db = await ClassifierDBProvider.get_taxonomy_db()
    cache_repo = EmbeddingsCacheRepository(app_db=app_db, taxonomy_db=taxonomy_db)
    embedding_svc = await get_embedding_service(nel_model_id)
    linking_repo = EntityLinkingRepository(cache_repo, dimensions=embedding_svc.dimensions)
    return NELService(
        linking_repository=linking_repo,
        cache_repository=cache_repo,
        embedding_service=embedding_svc,
    )


@router.post("", response_model=NELResponse)
async def link_entities(
    request: NELRequest,
    user_config: UserConfig = Depends(_get_user_config),
    svc: INELService = Depends(_get_service),
):
    from nel.config import DEFAULT_NEL_MODEL_ID, DEFAULT_TAXONOMY_MODEL_ID
    taxonomy_model_id = user_config.taxonomy_model_id or DEFAULT_TAXONOMY_MODEL_ID
    nel_model_id = user_config.nel_model_id or DEFAULT_NEL_MODEL_ID

    if not taxonomy_model_id:
        raise HTTPException(
            status_code=400,
            detail="No taxonomy model configured — set one via PUT /v2/nel/user/config or set DEFAULT_TAXONOMY_MODEL_ID",
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

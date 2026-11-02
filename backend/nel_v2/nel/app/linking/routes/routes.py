"""Entity linking routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from nel.app.embedding.service.service import (
    EmbeddingBackendUnavailableError,
    get_embedding_service,
)
from nel.app.embeddings_cache.repository.repository import EmbeddingsCacheRepository
from nel.app.linking.repository.repository import EntityLinkingRepository
from nel.app.linking.routes._types import LinkingModels, NELRequest
from nel.app.linking.service.errors import EmbeddingsCacheNotReadyError
from nel.app.linking.service.service import INELService, NELService
from nel.app.linking.service.types import NELOptions, NELResponse
from nel.app.server_dependencies.db_dependencies import ClassifierDBProvider
from nel.app.user_config.repository.repository import UserConfigRepository
from nel.app.user_config.routes.auth import get_firebase_uid
from nel.app.user_config.service.service import UserConfigService
from nel.app.user_config.service.types import UserConfig
from shared.languages import get_language_config, normalise_language

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/nel", tags=["nel"])


async def _get_user_config(uid: str = Depends(get_firebase_uid)) -> UserConfig:
    app_db = await ClassifierDBProvider.get_application_db()
    return await UserConfigService(UserConfigRepository(app_db)).get(uid)


async def _resolve_models(
    request: NELRequest, user_config: UserConfig = Depends(_get_user_config)
) -> LinkingModels:
    from nel.config import DEFAULT_NEL_MODEL_ID, DEFAULT_TAXONOMY_MODEL_ID

    # A language on the request is an explicit instruction, so its configured taxonomy
    # model wins over the caller's stored config. Without it, nothing changes.
    # The language's embedding model comes with it: embeddings are stored per
    # (taxonomy model, embedding model) pair, so the two are only used together.
    language_taxonomy_model_id = ""
    language_nel_model_id = ""
    if request.language:
        language = normalise_language(request.language)
        language_config = get_language_config(language)
        language_taxonomy_model_id = language_config.get("taxonomy_model_id") or ""
        if language_taxonomy_model_id:
            language_nel_model_id = language_config.get("nel_v2_model_id") or ""
        else:
            _logger.warning(
                "language=%s requested but TAXONOMY_MODEL_ID_%s is not set; "
                "falling back to the caller's taxonomy model",
                language,
                language.upper(),
            )

    taxonomy_model_id = (
        language_taxonomy_model_id or user_config.taxonomy_model_id or DEFAULT_TAXONOMY_MODEL_ID
    )
    if not taxonomy_model_id:
        raise HTTPException(
            status_code=400,
            detail="No taxonomy model configured — set one via PUT /v2/nel/user/config or set DEFAULT_TAXONOMY_MODEL_ID",
        )
    return LinkingModels(
        taxonomy_model_id=taxonomy_model_id,
        nel_model_id=language_nel_model_id or user_config.nel_model_id or DEFAULT_NEL_MODEL_ID,
    )


async def _get_service(models: LinkingModels = Depends(_resolve_models)) -> INELService:
    app_db = await ClassifierDBProvider.get_application_db()
    taxonomy_db = await ClassifierDBProvider.get_taxonomy_db()
    cache_repo = EmbeddingsCacheRepository(app_db=app_db, taxonomy_db=taxonomy_db)
    try:
        embedding_svc = await get_embedding_service(models.nel_model_id)
    except EmbeddingBackendUnavailableError as exc:
        # e.g. Vertex AI not enabled / no access on the project. This is an
        # operational/config problem, not a client error — surface it as 503
        # so the classify pipeline's NEL stage maps it to a retryable failure
        # rather than an opaque 500.
        raise HTTPException(status_code=503, detail=str(exc))
    linking_repo = EntityLinkingRepository(cache_repo, dimensions=embedding_svc.dimensions)
    return NELService(
        linking_repository=linking_repo,
        cache_repository=cache_repo,
        embedding_service=embedding_svc,
    )


@router.post("", response_model=NELResponse)
async def link_entities(
    request: NELRequest,
    models: LinkingModels = Depends(_resolve_models),
    svc: INELService = Depends(_get_service),
):
    try:
        return await svc.link(
            entities=[(e.text, e.entity_type) for e in request.entities],
            taxonomy_model_id=models.taxonomy_model_id,
            nel_model_id=models.nel_model_id,
            options=NELOptions(top_k=request.top_k, min_similarity=request.min_similarity),
        )
    except EmbeddingsCacheNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except EmbeddingBackendUnavailableError as exc:
        # Vertex became unavailable mid-request (embed call itself failed).
        raise HTTPException(status_code=503, detail=str(exc))

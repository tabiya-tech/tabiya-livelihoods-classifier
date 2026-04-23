"""Taxonomy models routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from nel.app.taxonomy_models.service.errors import TaxonomyAPIError
from nel.app.taxonomy_models.service.service import ITaxonomyModelService, TaxonomyModelService
from nel.app.taxonomy_models.service.types import TaxonomyModelInfo

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/nel/taxonomy-models", tags=["taxonomy-models"])


async def _get_service() -> ITaxonomyModelService:
    from nel.config import TAXONOMY_API_BASE_URL, TAXONOMY_API_KEY
    return TaxonomyModelService(base_url=TAXONOMY_API_BASE_URL, api_key=TAXONOMY_API_KEY)


@router.get("", response_model=list[TaxonomyModelInfo])
async def list_taxonomy_models(svc: ITaxonomyModelService = Depends(_get_service)):
    try:
        return await svc.get_all()
    except TaxonomyAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

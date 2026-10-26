"""Classify v2 routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from classify_v2.app.classification.service.errors import EmbeddingsCacheNotReadyError, NERServiceError, NELServiceError
from classify_v2.app.classification.service.service import ClassifyService, IClassifyService
from classify_v2.app.classification.service.types import ClassifyRequest, ClassifyResponse
from classify_v2.config import MAX_TEXT_LENGTH

_logger = logging.getLogger(__name__)

router = APIRouter(tags=["classify"])

# Headers set by the API Gateway that identify the end user: decoded Firebase
# claims, or the validated API key. nel_v2 resolves either one to the user and
# applies that user's model config, so they are forwarded on the NEL call.
_CALLER_IDENTITY_HEADER_NAMES = ("x-apigateway-api-userinfo", "x-api-key")


def _get_service() -> IClassifyService:
    return ClassifyService()


def _build_input_text(req: ClassifyRequest) -> str:
    if req.text:
        return req.text.strip()
    parts = [req.title or "", req.description or ""]
    return "\n".join(p.strip() for p in parts if p.strip())


def _extract_caller_identity_headers(http_request: Request) -> dict[str, str]:
    return {
        header_name: http_request.headers[header_name]
        for header_name in _CALLER_IDENTITY_HEADER_NAMES
        if http_request.headers.get(header_name)
    }


@router.post("/v2/classify", response_model=ClassifyResponse)
async def classify(
    request: ClassifyRequest,
    http_request: Request,
    svc: IClassifyService = Depends(_get_service),
):
    input_text = _build_input_text(request)
    if not input_text:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'title'+'description'")
    if len(input_text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=413, detail=f"Text exceeds maximum length ({MAX_TEXT_LENGTH} chars)")

    _logger.info("Classify v2 request: %d chars", len(input_text))
    try:
        result = await svc.classify(
            input_text,
            options=request.options,
            caller_identity_headers=_extract_caller_identity_headers(http_request),
        )
    except EmbeddingsCacheNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except (NERServiceError, NELServiceError) as exc:
        _logger.error("Classify v2 failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))

    _logger.info(
        "Classify v2 done: %d entities in %.1fms",
        len(result.entities),
        result.metadata.processing_time_ms,
    )
    return result

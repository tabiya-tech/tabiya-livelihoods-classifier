"""HTTP adapter helper — wraps a plugin's Core function in a FastAPI router.

Every bundle imports `make_http_adapter` and mounts the returned router under
`/plugin/{plugin_id}`. The helper handles:

  * Manifest emission (with `x-tabiya-contract-version` auto-injected).
  * Invoke request parsing — validates the `input` payload against the slot
    model resolved from the manifest's `input_slot.type`.
  * Timeout enforcement — hard `asyncio.wait_for` on `timeout_ms`.
  * Error mapping — turns `PluginError` subclasses into the six-value
    `ErrorCode` taxonomy with the correct HTTP status from design §1.
  * Health forwarding.

The Core function signature the adapter expects:

    async def invoke(
        input: InputSlotModel,
        config: dict[str, Any],
        context: Context,
    ) -> OutputSlotModel

Plugin authors do not touch this module directly; they call
`make_http_adapter(MANIFEST, invoke_fn, health_fn)` from their
`adapters/http.py` shim.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ..health import Health, HealthStatus
from ..invoke import Context, ErrorCode, ErrorEnvelope, InvokeRequest, InvokeResponse
from ..manifest import Manifest
from ..slots import SLOT_MODEL_BY_TYPE
from ..version import CONTRACT_VERSION

log = logging.getLogger("tabiya-plugin-adapter")


def _jsonable_validation_errors(exc: ValidationError) -> list[dict]:
    """Return ValidationError.errors() with non-JSON-safe values scrubbed.

    Pydantic embeds raw exceptions inside the `ctx` field for
    `value_error`-style validators, which the default JSON encoder can't
    handle. We surface `str(exc)` in their place so the envelope's
    `detail.errors` remains machine-readable.
    """

    cleaned: list[dict] = []
    for entry in exc.errors():
        entry_copy = dict(entry)
        ctx = entry_copy.get("ctx")
        if isinstance(ctx, dict):
            cleaned_ctx = {}
            for key, value in ctx.items():
                cleaned_ctx[key] = value if _is_json_safe(value) else str(value)
            entry_copy["ctx"] = cleaned_ctx
        cleaned.append(entry_copy)
    return cleaned


def _is_json_safe(value: object) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, dict, tuple))


class PluginError(Exception):
    """Base class for plugin errors mapped to the ErrorCode taxonomy."""

    code: ErrorCode = ErrorCode.PLUGIN_INTERNAL
    http_status: int = 500

    def __init__(self, message: str, detail: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class BadInputError(PluginError):
    code = ErrorCode.BAD_INPUT
    http_status = 400


class ConfigInvalidError(PluginError):
    code = ErrorCode.CONFIG_INVALID
    http_status = 400


class UpstreamUnavailableError(PluginError):
    code = ErrorCode.UPSTREAM_UNAVAILABLE
    http_status = 503


class UnavailableError(PluginError):
    code = ErrorCode.UNAVAILABLE
    http_status = 503


InvokeFn = Callable[[Any, dict[str, Any], Context], Awaitable[Any]]
HealthFn = Callable[[], Awaitable[Health]]


async def _default_health() -> Health:
    return Health(status=HealthStatus.OK)


def _error_response(err: PluginError) -> JSONResponse:
    envelope = ErrorEnvelope(code=err.code, message=err.message, detail=err.detail)
    return JSONResponse(status_code=err.http_status, content=envelope.model_dump())


def make_http_adapter(
    manifest: Manifest,
    invoke_fn: InvokeFn,
    health_fn: Optional[HealthFn] = None,
) -> APIRouter:
    """Return a FastAPI router mounting `/manifest`, `/invoke`, `/health`.

    The caller is expected to mount this under a per-plugin path prefix,
    e.g. `app.include_router(router, prefix=f"/plugin/{manifest.plugin_id}")`.
    """

    # Inject the contract version onto the manifest we serve so registries can
    # do compatibility checks without asking the plugin author to remember it.
    manifest_with_version = manifest.model_copy(
        update={"x_tabiya_contract_version": CONTRACT_VERSION}
    )

    input_slot_model = SLOT_MODEL_BY_TYPE[manifest.input_slot.type]
    output_slot_model = SLOT_MODEL_BY_TYPE[manifest.output_slot.type]
    health_impl = health_fn or _default_health

    router = APIRouter()

    @router.get("/manifest")
    async def get_manifest() -> dict[str, Any]:
        # `by_alias=True` preserves the `x-tabiya-*` field names as the wire spec.
        return manifest_with_version.model_dump(by_alias=True, exclude_none=True)

    @router.get("/health")
    async def get_health() -> dict[str, Any]:
        try:
            health = await health_impl()
        except Exception as exc:  # noqa: BLE001 — health must never crash the endpoint
            log.exception("Health check raised: %s", exc)
            return Health(status=HealthStatus.DOWN, detail=str(exc)).model_dump()
        return health.model_dump()

    @router.post("/invoke")
    async def post_invoke(request: Request) -> Any:
        try:
            raw_body = await request.json()
        except Exception:
            return _error_response(BadInputError("Request body must be valid JSON."))

        try:
            invoke_request = InvokeRequest.model_validate(raw_body)
        except ValidationError as exc:
            return _error_response(
                BadInputError(
                    "Invoke request envelope failed validation.",
                    detail={"errors": _jsonable_validation_errors(exc)},
                )
            )

        # Parse the typed input payload against the manifest's input slot.
        try:
            typed_input = input_slot_model.model_validate(invoke_request.input)
        except ValidationError as exc:
            return _error_response(
                BadInputError(
                    f"Input payload does not match slot type {manifest.input_slot.type.value}.",
                    detail={"errors": _jsonable_validation_errors(exc)},
                )
            )

        try:
            typed_output = await asyncio.wait_for(
                invoke_fn(typed_input, invoke_request.config, invoke_request.context),
                timeout=manifest.timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            envelope = ErrorEnvelope(
                code=ErrorCode.TIMEOUT,
                message=f"Plugin exceeded timeout of {manifest.timeout_ms}ms.",
            )
            return JSONResponse(status_code=504, content=envelope.model_dump())
        except PluginError as exc:
            return _error_response(exc)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 — turn anything else into 500 envelope
            log.exception("Plugin %s raised unhandled exception", manifest.plugin_id)
            envelope = ErrorEnvelope(
                code=ErrorCode.PLUGIN_INTERNAL,
                message=f"Unhandled plugin error: {exc}",
            )
            return JSONResponse(status_code=500, content=envelope.model_dump())

        # Validate that what the Core returned matches the declared output slot.
        try:
            validated_output = output_slot_model.model_validate(
                typed_output.model_dump() if hasattr(typed_output, "model_dump") else typed_output
            )
        except ValidationError as exc:
            envelope = ErrorEnvelope(
                code=ErrorCode.PLUGIN_INTERNAL,
                message=f"Plugin returned payload that does not match slot type {manifest.output_slot.type.value}.",
                detail={"errors": _jsonable_validation_errors(exc)},
            )
            return JSONResponse(status_code=500, content=envelope.model_dump())

        response = InvokeResponse(output=validated_output.model_dump())
        return response.model_dump()

    return router


__all__ = [
    "BadInputError",
    "ConfigInvalidError",
    "PluginError",
    "UnavailableError",
    "UpstreamUnavailableError",
    "jsonable_validation_errors",
    "make_http_adapter",
]


# Public alias so plugin authors can sanitise ValidationError before
# stuffing it into ConfigInvalidError.detail, without importing the
# underscore-prefixed private helper.
jsonable_validation_errors = _jsonable_validation_errors

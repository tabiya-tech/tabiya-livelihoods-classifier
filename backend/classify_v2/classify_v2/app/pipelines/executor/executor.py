"""Linear pipeline executor.

Walks a `PipelineDocument`'s stages in order, POSTing each stage's typed
`input` payload to the plugin bundle's `/invoke` endpoint. The output of
stage N becomes the input of stage N+1.

The executor:
  * Resolves each plugin_id via the registry (raising the same errors
    the validator would surface at save time if we're now stale).
  * Enforces per-stage timeouts from `manifest.timeout_ms`.
  * Attaches a GCP identity token when running against real bundle URLs
    (skipped in local mode / when the metadata server is unreachable).
  * Records a `StageOutcome` per stage — the caller uses these both to
    build the classify response `metadata.pipeline` block and to feed
    the observability hooks in 11.6b.

The executor does NOT know about the shape of any particular slot
payload (RawText vs Entities vs LinkedEntities). Everything travels as
`dict[str, Any]` between stages; only the Source config override and the
final-stage extraction (see `_extract_linked_entities`) touch shape.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from classify_v2.app.pipelines.executor.errors import (
    PluginInvocationError,
    PluginTimeoutError,
    PluginUpstreamUnavailableError,
)
from classify_v2.app.pipelines.registry import (
    PluginRegistry,
    PluginStatus,
    PluginUnreachableError,
)
from classify_v2.app.pipelines.repository import PipelineDocument, StageDocument
from tabiya_plugin_contracts import CONTRACT_VERSION, Manifest, SlotType

_logger = logging.getLogger(__name__)

# Dedicated logger for the observability stream — ops dashboards should filter on
# `logger.name == "classify_v2.pipeline_stage"` and read the extras as
# structured JSON (Cloud Run's log agent lifts `extra=` fields into
# `jsonPayload` on every entry).
_stage_logger = logging.getLogger("classify_v2.pipeline_stage")

_NEL_PLUGIN_ID = "tabiya.nel.v1"


def _log_stage(
    *,
    pipeline_id: str,
    stage_index: int,
    plugin_id: str,
    plugin_version: str,
    category: str,
    duration_ms: float,
    status: str,
    error_code: Optional[str] = None,
    request_id: Optional[str] = None,
) -> None:
    """Emit one structured log line per plugin /invoke call.

    Do not add or rename fields here without also updating the ops dashboard queries.
    """

    extra: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "stage_index": stage_index,
        "plugin_id": plugin_id,
        "plugin_version": plugin_version,
        "category": category,
        "duration_ms": round(duration_ms, 3),
        "status": status,
        "request_id": request_id,
    }
    if error_code is not None:
        extra["error_code"] = error_code
    _stage_logger.info(
        "stage %d %s %s in %.1fms",
        stage_index,
        plugin_id,
        status,
        duration_ms,
        extra=extra,
    )


@dataclass
class StageOutcome:
    """One row of the executor's per-stage trace.

    `duration_ms` is wall-clock time from just before the /invoke POST to
    just after the response body is fully parsed.
    """

    stage_index: int
    plugin_id: str
    plugin_version: str
    category: str
    duration_ms: float
    status: str  # "ok" | "timeout" | "error"
    error_code: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class ExecutorResult:
    """The executor's full output.

    `final_output` is the payload from the last stage (which for the
    canonical `text_input → ner → nel → results` pipeline is `None` since
    the sink's output slot is None).

    `linked_entities_payload` is the last LinkedEntities payload observed
    in the pipeline — the classify route uses this to build the response
    body. Held separately from `final_output` so a Sink can still return
    None without discarding the classify caller's intended result.
    """

    pipeline_id: str
    pipeline_name: str
    stages: list[StageOutcome]
    final_output: Optional[dict[str, Any]]
    linked_entities_payload: Optional[dict[str, Any]]


class PipelineExecutor:
    """Executes a persisted pipeline document over live plugin bundles."""

    def __init__(
        self,
        *,
        registry: PluginRegistry,
        http_client: httpx.AsyncClient,
        identity_token_provider: Optional[Any] = None,
    ) -> None:
        self._registry = registry
        self._http = http_client
        self._identity = identity_token_provider

    async def run(  # noqa: PLR0915 — logging is inline for readability
        self,
        *,
        pipeline: PipelineDocument,
        source_overrides: Optional[dict[str, Any]] = None,
        stage_config_overrides: Optional[dict[int, dict[str, Any]]] = None,
        request_id: str,
        user_id: Optional[str],
    ) -> ExecutorResult:
        """Run the pipeline end-to-end.

        `source_overrides` is merged into the Source stage's config right
        before invoking — the classify route passes `{text: "..."}` (or
        `{title, description}`) here so the persisted default is
        overridden by the caller's per-request input.

        Emits one structured log line per stage on
        `classify_v2.pipeline_stage` regardless of outcome.
        """

        outcomes: list[StageOutcome] = []
        linked_entities_payload: Optional[dict[str, Any]] = None
        current_input: dict[str, Any] = {"kind": "None"}  # NoneSlot for Source

        for stage_index, stage in enumerate(pipeline.stages):
            try:
                manifest = self._resolve_manifest(stage, stage_index)
            except PluginInvocationError as exc:
                # Fall through to the log line below with a placeholder
                # version/category since we don't know them without a
                # manifest — then re-raise.
                _log_stage(
                    pipeline_id=pipeline.pipeline_id,
                    stage_index=stage_index,
                    plugin_id=stage.plugin_id,
                    plugin_version="unknown",
                    category="unknown",
                    duration_ms=0.0,
                    status="error",
                    error_code="UNAVAILABLE",
                    request_id=request_id,
                )
                raise

            resolved_url = self._resolved_url(stage.plugin_id)

            stage_config: dict[str, Any] = {**(stage.config or {})}
            if stage_index == 0 and source_overrides:
                stage_config.update(source_overrides)
            if stage_config_overrides and stage_index in stage_config_overrides:
                stage_config.update(stage_config_overrides[stage_index])

            invoke_body = {
                "context": {
                    "request_id": request_id,
                    "user_id": user_id,
                    "pipeline_id": pipeline.pipeline_id,
                    "stage_index": stage_index,
                    "deadline_ms": manifest.timeout_ms,
                },
                "config": stage_config,
                "input": current_input,
            }
            invoke_url = f"{resolved_url}/invoke"

            outcome, response_body = await self._invoke_one(
                stage_index=stage_index,
                plugin_id=stage.plugin_id,
                manifest=manifest,
                invoke_url=invoke_url,
                body=invoke_body,
                pipeline_id=pipeline.pipeline_id,
                request_id=request_id,
            )
            outcomes.append(outcome)

            current_input = response_body.get("output") or {"kind": "None"}
            # Snapshot the entities payload for the response. Prefer the last
            # LinkedEntities we see (canonical text→ner→nel→results). If the
            # pipeline ends on NER with no NEL stage, capture the Entities
            # payload instead — its `matches` are simply empty. Entities and
            # LinkedEntities share shape, so the response builder handles both.
            if manifest.output_slot.type == SlotType.LINKED_ENTITIES:
                linked_entities_payload = current_input
            elif (
                manifest.output_slot.type == SlotType.ENTITIES
                and linked_entities_payload is None
            ):
                linked_entities_payload = current_input

        return ExecutorResult(
            pipeline_id=pipeline.pipeline_id,
            pipeline_name=pipeline.name,
            stages=outcomes,
            final_output=current_input,
            linked_entities_payload=linked_entities_payload,
        )

    # ── stage invocation ───────────────────────────────────────────────

    async def _invoke_one(
        self,
        *,
        stage_index: int,
        plugin_id: str,
        manifest: Manifest,
        invoke_url: str,
        body: dict[str, Any],
        pipeline_id: str,
        request_id: str,
    ) -> tuple[StageOutcome, dict[str, Any]]:
        started = time.monotonic()

        def _emit_log(status: str, error_code: Optional[str] = None, *, duration_ms: float) -> None:
            _log_stage(
                pipeline_id=pipeline_id,
                stage_index=stage_index,
                plugin_id=plugin_id,
                plugin_version=manifest.version,
                category=manifest.category.value,
                duration_ms=duration_ms,
                status=status,
                error_code=error_code,
                request_id=request_id,
            )

        headers = await self._auth_headers(invoke_url)
        timeout_seconds = max(manifest.timeout_ms / 1000.0, 0.1)
        try:
            response = await self._http.post(
                invoke_url,
                json=body,
                headers=headers,
                timeout=timeout_seconds,
            )
        except (httpx.TimeoutException, asyncio.TimeoutError) as exc:
            duration_ms = (time.monotonic() - started) * 1000.0
            _emit_log("timeout", "TIMEOUT", duration_ms=duration_ms)
            raise PluginTimeoutError(
                f"Plugin '{plugin_id}' timed out after {manifest.timeout_ms}ms.",
                stage_index=stage_index,
                plugin_id=plugin_id,
            ) from exc
        except httpx.RequestError as exc:
            duration_ms = (time.monotonic() - started) * 1000.0
            _emit_log("error", "UNAVAILABLE", duration_ms=duration_ms)
            raise PluginInvocationError(
                f"Plugin '{plugin_id}' unreachable: {exc}",
                stage_index=stage_index,
                plugin_id=plugin_id,
            ) from exc

        duration_ms = (time.monotonic() - started) * 1000.0

        if response.status_code == 503:
            envelope = _safe_json(response)
            _emit_log("error", "UPSTREAM_UNAVAILABLE", duration_ms=duration_ms)
            raise PluginUpstreamUnavailableError(
                envelope.get("message") if isinstance(envelope, dict) else f"Plugin '{plugin_id}' upstream unavailable.",
                stage_index=stage_index,
                plugin_id=plugin_id,
                detail=envelope if isinstance(envelope, dict) else None,
            )

        if response.status_code == 504:
            envelope = _safe_json(response)
            _emit_log("timeout", "TIMEOUT", duration_ms=duration_ms)
            raise PluginTimeoutError(
                envelope.get("message") if isinstance(envelope, dict) else f"Plugin '{plugin_id}' reported timeout.",
                stage_index=stage_index,
                plugin_id=plugin_id,
                detail=envelope if isinstance(envelope, dict) else None,
            )

        if response.status_code != 200:
            envelope = _safe_json(response)
            error_code = None
            message = f"Plugin '{plugin_id}' returned HTTP {response.status_code}."
            if isinstance(envelope, dict):
                error_code = envelope.get("code")
                message = envelope.get("message") or message
            _emit_log("error", error_code or "PLUGIN_INTERNAL", duration_ms=duration_ms)
            raise PluginInvocationError(
                message,
                stage_index=stage_index,
                plugin_id=plugin_id,
                detail={
                    "http_status": response.status_code,
                    "error_code": error_code,
                    "envelope": envelope if isinstance(envelope, dict) else None,
                },
            )

        try:
            response_body = response.json()
        except ValueError as exc:
            _emit_log("error", "PLUGIN_INTERNAL", duration_ms=duration_ms)
            raise PluginInvocationError(
                f"Plugin '{plugin_id}' returned invalid JSON: {exc}",
                stage_index=stage_index,
                plugin_id=plugin_id,
            ) from exc

        outcome = StageOutcome(
            stage_index=stage_index,
            plugin_id=plugin_id,
            plugin_version=manifest.version,
            category=manifest.category.value,
            duration_ms=duration_ms,
            status="ok",
            metadata=response_body.get("metadata"),
        )
        _emit_log("ok", duration_ms=duration_ms)
        return outcome, response_body

    # ── helpers ────────────────────────────────────────────────────────

    def _resolve_manifest(self, stage: StageDocument, stage_index: int) -> Manifest:
        resolved = self._registry.get(stage.plugin_id)
        if resolved is None:
            raise PluginInvocationError(
                f"Plugin '{stage.plugin_id}' is not in the catalog.",
                stage_index=stage_index,
                plugin_id=stage.plugin_id,
            )
        if resolved.coming_soon:
            raise PluginInvocationError(
                f"Plugin '{stage.plugin_id}' is marked Coming Soon and cannot be invoked.",
                stage_index=stage_index,
                plugin_id=stage.plugin_id,
            )
        if resolved.status != PluginStatus.ENABLED or resolved.manifest is None:
            raise PluginInvocationError(
                f"Plugin '{stage.plugin_id}' is currently unavailable ({resolved.status.value}).",
                stage_index=stage_index,
                plugin_id=stage.plugin_id,
                detail={"last_error": resolved.last_error} if resolved.last_error else None,
            )
        return resolved.manifest

    def _resolved_url(self, plugin_id: str) -> str:
        try:
            resolved = self._registry.get(plugin_id)
        except PluginUnreachableError:
            resolved = None
        if resolved is None or resolved.resolved_url is None:
            # Should be unreachable — _resolve_manifest above already
            # asserted ENABLED. This is defensive so a race between
            # refresh and invoke fails loudly rather than posting to None.
            raise PluginInvocationError(
                f"Plugin '{plugin_id}' has no resolved URL.",
                stage_index=-1,
                plugin_id=plugin_id,
            )
        return resolved.resolved_url

    async def _auth_headers(self, url: str) -> dict[str, str]:
        headers = {"x-tabiya-contract-version": CONTRACT_VERSION}
        if self._identity is None:
            return headers
        try:
            token = await self._identity.get_id_token(url)
        except Exception:  # noqa: BLE001 — auth is best-effort; local mode has no token
            _logger.debug("Identity token unavailable for %s", url, exc_info=True)
            return headers
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None

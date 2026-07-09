"""Read-only plugin catalog endpoints.

GET  /v2/plugins                              — palette summaries
GET  /v2/plugins/{plugin_id}                  — full manifest + status
GET  /v2/plugins/{plugin_id}/options/{field}  — proxies the manifest's
                                                  `x-source` for a config field

All three require an authenticated user (Firebase or gateway API key).
Local mode bypasses auth via `TARGET_ENVIRONMENT_TYPE=local` as per
existing behaviour.

The options endpoint is a thin proxy: the plugin's manifest declares
where the dropdown data lives (`config_schema.properties.<field>.x-source`),
this route resolves the URL against classify_v2's own base and forwards
the JSON verbatim to the frontend. That keeps the frontend from having
to know about the shape of every plugin's options space.
"""

from __future__ import annotations

import ipaddress
import logging
import urllib.parse
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status

from classify_v2.app.auth.firebase import get_firebase_uid
from classify_v2.app.pipelines.plugins_routes._types import (
    ListPluginsResponse,
    PluginDetail,
    PluginOptionItem,
    PluginOptionsResponse,
    PluginSummary,
)
from classify_v2.app.pipelines.registry import (
    PluginRegistry,
    PluginStatus,
    PluginUnreachableError,
    ResolvedPlugin,
)

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/plugins", tags=["plugins"])


def get_plugin_registry(request: Request) -> PluginRegistry:
    """Read the registry singleton set on app.state during lifespan."""

    registry = getattr(request.app.state, "plugin_registry", None)
    if registry is None:
        raise RuntimeError(
            "Plugin registry was not initialised during app startup; "
            "check classify_v2.main.lifespan"
        )
    return registry


def get_plugin_http(request: Request) -> httpx.AsyncClient:
    client = getattr(request.app.state, "plugin_http", None)
    if client is None:
        raise RuntimeError(
            "Plugin HTTP client was not initialised during app startup; "
            "check classify_v2.main.lifespan"
        )
    return client


def _summary_from(resolved: ResolvedPlugin) -> PluginSummary:
    manifest = resolved.manifest
    return PluginSummary(
        plugin_id=resolved.plugin_id,
        name=manifest.name if manifest else resolved.plugin_id,
        version=manifest.version if manifest else "",
        category=manifest.category if manifest else None,
        summary=manifest.summary if manifest else "",
        detail=manifest.detail if manifest else None,
        icon=manifest.icon if manifest else "",
        status=resolved.status,
        coming_soon=resolved.coming_soon,
        last_error=resolved.last_error,
    )


@router.get("", response_model=ListPluginsResponse)
async def list_plugins(
    _uid: str = Depends(get_firebase_uid),
    registry: PluginRegistry = Depends(get_plugin_registry),
) -> ListPluginsResponse:
    return ListPluginsResponse(
        plugins=[_summary_from(entry) for entry in registry.list_manifests()],
    )


@router.get("/{plugin_id}", response_model=PluginDetail)
async def get_plugin(
    plugin_id: str,
    _uid: str = Depends(get_firebase_uid),
    registry: PluginRegistry = Depends(get_plugin_registry),
) -> PluginDetail:
    resolved = registry.get(plugin_id)
    if resolved is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plugin not found")
    return PluginDetail(
        plugin_id=resolved.plugin_id,
        status=resolved.status,
        coming_soon=resolved.coming_soon,
        last_error=resolved.last_error,
        manifest=resolved.manifest,
    )


def _find_x_source(manifest: Any, field: str) -> str | None:
    """Return the `x-source` URL declared for `field` in the manifest.

    We deliberately walk `config_schema` as raw dict data rather than
    binding it to a Pydantic model here — plugin authors write these
    schemas by hand and Pydantic would strip unknown keys.
    """

    schema = manifest.config_schema if manifest is not None else None
    if not isinstance(schema, dict):
        return None
    props = schema.get("properties")
    if not isinstance(props, dict):
        return None
    field_schema = props.get(field)
    if not isinstance(field_schema, dict):
        return None
    x_source = field_schema.get("x-source")
    return x_source if isinstance(x_source, str) and x_source else None


def _is_private_host(host: str) -> bool:
    """Return True if `host` resolves to a loopback or RFC-1918 address.

    We check the literal hostname only (no DNS resolution) so that
    cloud metadata endpoints like `169.254.169.254` are blocked without
    a network round-trip.
    """
    try:
        addr = ipaddress.ip_address(host)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except ValueError:
        # Not a bare IP — allow it (hostname like `nel-service.internal` is
        # fine; DNS-rebinding is an accepted residual risk here because
        # x-source values come from operator-configured plugin bundles, not
        # from end-user input).
        return False


def _resolve_options_url(request: Request, x_source: str) -> str:
    """Turn a relative x-source (`/v2/nel/models`) into an absolute URL.

    Absolute URLs pass through after a private-IP check. Relative paths
    resolve against this request's own base_url so that `/v2/nel/*` routes
    proxy to sibling services in the same deployment.
    """

    if x_source.startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(x_source)
        if _is_private_host(parsed.hostname or ""):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Plugin x-source URL '{x_source}' targets a private or "
                    "loopback address, which is not permitted."
                ),
            )
        return x_source
    base = str(request.base_url).rstrip("/")
    if not x_source.startswith("/"):
        x_source = "/" + x_source
    return f"{base}{x_source}"


def _coerce_options_payload(payload: Any) -> list[PluginOptionItem]:
    """Turn the upstream JSON into a list of `{value, label}` items.

    The upstream endpoint can return:
      * `[{"value": "...", "label": "..."}]` — already-shaped, easy case.
      * `[{"id": "...", "name": "..."}]` — the current /v2/nel/models shape.
      * `[{"model_id": "...", "display_name": "..."}]` — near-equivalent.
      * A list of strings.

    Anything else raises to give the frontend a clean 502.
    """

    if not isinstance(payload, list):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Options upstream did not return a JSON list.",
        )
    items: list[PluginOptionItem] = []
    for entry in payload:
        if isinstance(entry, str):
            items.append(PluginOptionItem(value=entry, label=entry))
            continue
        if not isinstance(entry, dict):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Options upstream returned a non-object entry.",
            )
        value = (
            entry.get("value")
            or entry.get("id")
            or entry.get("model_id")
            or entry.get("taxonomy_model_id")
        )
        label = (
            entry.get("label")
            or entry.get("name")
            or entry.get("display_name")
            or value
        )
        if value is None:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Options upstream entry has no recognisable id field.",
            )
        items.append(PluginOptionItem(value=str(value), label=str(label)))
    return items


@router.get("/{plugin_id}/options/{field}", response_model=PluginOptionsResponse)
async def get_plugin_options(
    plugin_id: str,
    field: str,
    request: Request,
    _uid: str = Depends(get_firebase_uid),
    registry: PluginRegistry = Depends(get_plugin_registry),
    http: httpx.AsyncClient = Depends(get_plugin_http),
) -> PluginOptionsResponse:
    resolved = registry.get(plugin_id)
    if resolved is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plugin not found")
    if resolved.status != PluginStatus.ENABLED:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Plugin '{plugin_id}' is not currently available "
                f"({resolved.status.value})."
            ),
        )
    try:
        manifest = registry.get_manifest(plugin_id)
    except PluginUnreachableError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        )

    x_source = _find_x_source(manifest, field)
    if x_source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Plugin '{plugin_id}' has no `x-source` for field '{field}'."
            ),
        )

    upstream_url = _resolve_options_url(request, x_source)
    try:
        upstream_response = await http.get(upstream_url, timeout=5.0)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Options upstream unreachable: {exc}",
        )
    if upstream_response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Options upstream returned HTTP {upstream_response.status_code}."
            ),
        )
    try:
        payload = upstream_response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Options upstream returned invalid JSON: {exc}",
        )
    return PluginOptionsResponse(
        field=field, options=_coerce_options_payload(payload)
    )

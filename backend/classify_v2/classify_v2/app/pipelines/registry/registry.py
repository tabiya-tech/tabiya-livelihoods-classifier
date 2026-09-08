"""Plugin registry.

Resolves each catalog entry's URL from env vars and fetches its
`/plugin/manifest` on first use (lazy). The manifest is cached for the
lifetime of the process — manual invalidation happens automatically on
redeploy since classify_v2 restarts and the in-memory cache is cleared.

The registry is the single source of truth for two questions:

  * "Which plugin ids exist and where do they live?"  — `list_manifests()`,
    `get_manifest(plugin_id)`.
  * "Is this plugin currently reachable?"             — `get_status(plugin_id)`.

Nothing else in classify_v2 should hit `/plugin/manifest` directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

import httpx
from pydantic import ValidationError
from tabiya_plugin_contracts import CONTRACT_VERSION, Manifest

from .errors import PluginManifestInvalidError, PluginUnreachableError
from .types import CatalogEntry, PluginStatus, ResolvedPlugin

_logger = logging.getLogger(__name__)

DEFAULT_CATALOG_PATH = Path(__file__).parent / "catalog.json"
DEFAULT_FETCH_TIMEOUT_SECONDS = 30.0


class IHttpClient(Protocol):
    """The subset of httpx.AsyncClient the registry needs.

    Kept as a Protocol so tests can inject a deterministic fake without
    touching the network layer. In production the real AsyncClient is
    passed at PluginRegistry construction.
    """

    async def get(
        self,
        url: str,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response: ...


def load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> list[CatalogEntry]:
    """Parse the JSON catalog and validate each entry.

    Called once at registry construction. If the JSON is malformed or an
    entry doesn't match `CatalogEntry`, this raises immediately — a
    misconfigured catalog is a deployment bug we want to fail fast on.
    """

    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"catalog.json must be a JSON list, got {type(raw).__name__}")
    return [CatalogEntry.model_validate(entry) for entry in raw]


def _contract_major(version: str) -> str:
    return version.split(".", 1)[0]


class PluginRegistry:
    """In-memory cache of resolved plugins.

    Not thread-safe; assumed to be accessed from a single event loop.
    """

    def __init__(
        self,
        catalog: list[CatalogEntry],
        *,
        http_client: IHttpClient,
        env: Optional[dict[str, str]] = None,
        fetch_timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
        identity_token_provider: Optional[Any] = None,
    ) -> None:
        self._catalog = catalog
        self._http = http_client
        self._env = dict(env) if env is not None else dict(os.environ)
        self._fetch_timeout = fetch_timeout_seconds
        # Attaches a GCP identity token to manifest fetches so a private
        # Cloud Run bundle accepts them. None in local mode / tests → no header.
        self._identity = identity_token_provider
        self._plugins: dict[str, ResolvedPlugin] = {
            entry.plugin_id: ResolvedPlugin(
                plugin_id=entry.plugin_id,
                coming_soon=entry.coming_soon,
                status=PluginStatus.UNAVAILABLE,
            )
            for entry in catalog
        }
        # Per-plugin locks prevent concurrent first-requests from racing to
        # fetch the same manifest simultaneously.
        self._fetch_locks: dict[str, asyncio.Lock] = {
            entry.plugin_id: asyncio.Lock() for entry in catalog
        }

    def _resolve_url(self, entry: CatalogEntry) -> Optional[str]:
        if entry.coming_soon:
            return None
        base = self._env.get(entry.url_env, "").rstrip("/")
        if not base:
            return None
        return base + entry.path

    async def _auth_headers(self, url: str) -> dict[str, str]:
        """Attach a GCP identity token for `url` when a provider is set.

        No-op (empty dict) in local mode / tests, where the bundles bypass
        auth and no provider is configured.
        """

        if self._identity is None:
            return {}
        try:
            token = await self._identity.get_id_token(url)
        except Exception:  # noqa: BLE001 — auth is best-effort at manifest-fetch time
            _logger.debug("Identity token unavailable for %s", url, exc_info=True)
            return {}
        return {"Authorization": f"Bearer {token}"} if token else {}

    async def refresh(self) -> None:
        """Fetch (or re-fetch) every manifest once.

        Called by the smoke-test health endpoint and by tests. Runs
        sequentially so ordering in logs is stable.
        """

        for entry in self._catalog:
            await self._refresh_one(entry)

    async def _refresh_one(self, entry: CatalogEntry) -> None:
        plugin_id = entry.plugin_id
        current = self._plugins[plugin_id]

        if entry.coming_soon:
            self._plugins[plugin_id] = current.model_copy(
                update={
                    "resolved_url": None,
                    "manifest": None,
                    "status": PluginStatus.UNAVAILABLE,
                    "last_error": "coming_soon",
                    "last_refreshed_at": datetime.now(timezone.utc),
                }
            )
            return

        url = self._resolve_url(entry)
        if url is None:
            self._plugins[plugin_id] = current.model_copy(
                update={
                    "resolved_url": None,
                    "manifest": None,
                    "status": PluginStatus.UNAVAILABLE,
                    "last_error": f"env var {entry.url_env} is unset",
                    "last_refreshed_at": datetime.now(timezone.utc),
                }
            )
            _logger.warning(
                "Plugin '%s' UNAVAILABLE: env var %s is unset",
                plugin_id,
                entry.url_env,
            )
            return

        manifest_url = f"{url}/manifest"
        headers = await self._auth_headers(url)
        try:
            response = await self._http.get(
                manifest_url, timeout=self._fetch_timeout, headers=headers
            )
        except (httpx.RequestError, asyncio.TimeoutError) as exc:
            self._plugins[plugin_id] = current.model_copy(
                update={
                    "resolved_url": url,
                    "manifest": None,
                    "status": PluginStatus.UNAVAILABLE,
                    "last_error": f"unreachable: {exc}",
                    "last_refreshed_at": datetime.now(timezone.utc),
                }
            )
            _logger.warning(
                "Plugin '%s' UNAVAILABLE (unreachable): %s", plugin_id, exc
            )
            return

        if response.status_code != 200:
            self._plugins[plugin_id] = current.model_copy(
                update={
                    "resolved_url": url,
                    "manifest": None,
                    "status": PluginStatus.UNAVAILABLE,
                    "last_error": f"HTTP {response.status_code} from /plugin/manifest",
                    "last_refreshed_at": datetime.now(timezone.utc),
                }
            )
            _logger.warning(
                "Plugin '%s' UNAVAILABLE: /plugin/manifest returned %s",
                plugin_id,
                response.status_code,
            )
            return

        try:
            manifest = Manifest.model_validate(response.json())
        except (ValidationError, ValueError) as exc:
            self._plugins[plugin_id] = current.model_copy(
                update={
                    "resolved_url": url,
                    "manifest": None,
                    "status": PluginStatus.UNAVAILABLE,
                    "last_error": f"invalid manifest: {exc}",
                    "last_refreshed_at": datetime.now(timezone.utc),
                }
            )
            _logger.error("Plugin '%s' UNAVAILABLE: invalid manifest: %s", plugin_id, exc)
            return

        if manifest.plugin_id != plugin_id:
            self._plugins[plugin_id] = current.model_copy(
                update={
                    "resolved_url": url,
                    "manifest": None,
                    "status": PluginStatus.UNAVAILABLE,
                    "last_error": (
                        f"plugin_id mismatch: catalog says '{plugin_id}', "
                        f"manifest says '{manifest.plugin_id}'"
                    ),
                    "last_refreshed_at": datetime.now(timezone.utc),
                }
            )
            _logger.error(
                "Plugin '%s' UNAVAILABLE: plugin_id mismatch — manifest declares '%s'",
                plugin_id,
                manifest.plugin_id,
            )
            return

        if manifest.x_tabiya_contract_version is not None:
            plugin_major = _contract_major(manifest.x_tabiya_contract_version)
            our_major = _contract_major(CONTRACT_VERSION)
            if plugin_major != our_major:
                self._plugins[plugin_id] = current.model_copy(
                    update={
                        "resolved_url": url,
                        "manifest": None,
                        "status": PluginStatus.UNAVAILABLE,
                        "last_error": (
                            f"contract-version major mismatch: plugin uses "
                            f"{manifest.x_tabiya_contract_version}, orchestrator "
                            f"uses {CONTRACT_VERSION}. Rebuild required."
                        ),
                        "last_refreshed_at": datetime.now(timezone.utc),
                    }
                )
                _logger.error(
                    "Plugin '%s' UNAVAILABLE: contract-version mismatch "
                    "(plugin=%s, orchestrator=%s)",
                    plugin_id,
                    manifest.x_tabiya_contract_version,
                    CONTRACT_VERSION,
                )
                return

        # A plugin may ship a real manifest while declaring itself not-yet-
        # implemented via `x-tabiya-coming-soon`. Cache the manifest (so the
        # palette shows its name/category/icon) but keep it UNAVAILABLE +
        # coming_soon so it stays undroppable and the validator rejects it.
        is_coming_soon = bool(manifest.x_tabiya_coming_soon) or entry.coming_soon
        self._plugins[plugin_id] = current.model_copy(
            update={
                "resolved_url": url,
                "manifest": manifest,
                "status": (
                    PluginStatus.UNAVAILABLE if is_coming_soon else PluginStatus.ENABLED
                ),
                "coming_soon": is_coming_soon,
                "last_error": "coming_soon" if is_coming_soon else None,
                "last_refreshed_at": datetime.now(timezone.utc),
            }
        )
        if is_coming_soon:
            _logger.info("Plugin '%s' COMING SOON (manifest served) at %s", plugin_id, url)
        else:
            _logger.info("Plugin '%s' ENABLED at %s", plugin_id, url)

    # ── Read API ────────────────────────────────────────────────────────

    def list_manifests(self) -> list[ResolvedPlugin]:
        return list(self._plugins.values())

    async def get(self, plugin_id: str) -> Optional[ResolvedPlugin]:
        """Return the resolved plugin, fetching its manifest on first access.

        Thread-safe for concurrent coroutines: a per-plugin lock ensures only
        one fetch runs at a time; subsequent callers return the cached result.
        """

        plugin = self._plugins.get(plugin_id)
        if plugin is None:
            return None

        # Already successfully loaded — return the cached manifest.
        if plugin.status == PluginStatus.ENABLED:
            return plugin

        # Coming-soon plugins never become available — no point retrying.
        if plugin.coming_soon:
            return plugin

        lock = self._fetch_locks.get(plugin_id)
        if lock is None:
            return plugin

        async with lock:
            # Re-check inside the lock in case another coroutine just loaded it.
            if self._plugins[plugin_id].status == PluginStatus.ENABLED:
                return self._plugins[plugin_id]
            entry = next(e for e in self._catalog if e.plugin_id == plugin_id)
            await self._refresh_one(entry)

        return self._plugins[plugin_id]

    async def get_manifest(self, plugin_id: str) -> Manifest:
        """Return the manifest, fetching it lazily on first call.

        Raises PluginUnreachableError if the plugin isn't in the catalog or
        its manifest fetch failed.
        """

        entry = await self.get(plugin_id)
        if entry is None:
            raise PluginUnreachableError(
                plugin_id=plugin_id, url="<unknown>", reason="not in catalog"
            )
        if entry.manifest is None:
            raise PluginUnreachableError(
                plugin_id=plugin_id,
                url=entry.resolved_url or "<unset>",
                reason=entry.last_error or "manifest not loaded",
            )
        return entry.manifest

    async def get_status(self, plugin_id: str) -> PluginStatus:
        plugin = self._plugins.get(plugin_id)
        return plugin.status if plugin is not None else PluginStatus.UNAVAILABLE

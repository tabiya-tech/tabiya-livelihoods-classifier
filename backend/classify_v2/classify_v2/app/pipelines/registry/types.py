"""Typed value objects for the plugin registry.

`CatalogEntry` mirrors the JSON shape in `catalog.json`.
`ResolvedPlugin` is what the registry hands back to callers — a catalog
entry that has been through URL resolution + manifest fetch.
`PluginStatus` covers the three states from design §10.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from tabiya_plugin_contracts import Manifest


class PluginStatus(str, Enum):
    ENABLED = "enabled"
    """URL resolved, last manifest fetch succeeded, health last poll was healthy."""

    DEGRADED = "degraded"
    """URL resolved, manifest cached, but last health poll returned non-ok."""

    UNAVAILABLE = "unavailable"
    """No URL configured, or the manifest fetch failed / contract-version mismatch.

    Also used for `coming_soon` catalog entries — they're deliberately not
    reachable but must appear in the catalog so the frontend can render
    them as "Coming soon" cards.
    """


class CatalogEntry(BaseModel):
    """Raw catalog entry as it appears in catalog.json.

    Two shapes accepted (mirrors the JSON):
      * `{"plugin_id": "...", "url_env": "...", "path": "..."}` — normal case.
      * `{"plugin_id": "...", "coming_soon": true}` — placeholder for plugins
        that appear in the palette but are not deployable yet.
    """

    plugin_id: str
    url_env: str = ""
    path: str = ""
    coming_soon: bool = False


class ResolvedPlugin(BaseModel):
    """Registry-side view of a plugin.

    `manifest` is populated once the registry successfully fetches from
    `resolved_url`. `status` may transition ENABLED → DEGRADED between
    refreshes if the plugin's health starts failing.
    """

    plugin_id: str
    resolved_url: Optional[str] = Field(
        default=None,
        description="`{env[url_env]}{path}` or None when url_env is unset / coming_soon.",
    )
    manifest: Optional[Manifest] = None
    status: PluginStatus = PluginStatus.UNAVAILABLE
    coming_soon: bool = False
    last_error: Optional[str] = Field(
        default=None,
        description="Human-readable reason the plugin is UNAVAILABLE or DEGRADED.",
    )
    last_refreshed_at: Optional[datetime] = None

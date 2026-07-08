"""Response types for /v2/plugins.

`PluginSummary` powers the palette; `PluginDetail` powers the manifest
detail view + editor's ConfigForm. Both carry the runtime status so the
frontend can render UNAVAILABLE plugins as greyed-out cards.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from tabiya_plugin_contracts import Manifest, PluginCategory

from classify_v2.app.pipelines.registry import PluginStatus


class PluginSummary(BaseModel):
    """Palette-sized manifest projection.

    Field selection matches what the frontend Storybook stories consume:
    icon + name + category + one-line summary + runtime status.
    """

    plugin_id: str
    name: str = Field(default="")
    version: str = Field(default="")
    category: Optional[PluginCategory] = None
    summary: str = Field(default="")
    detail: Optional[str] = None
    icon: str = Field(default="")
    status: PluginStatus
    coming_soon: bool = False
    last_error: Optional[str] = None


class ListPluginsResponse(BaseModel):
    plugins: list[PluginSummary]


class PluginDetail(BaseModel):
    """Full manifest + runtime status.

    `manifest` is `None` for `coming_soon` / `UNAVAILABLE` plugins whose
    manifest hasn't been successfully fetched. The frontend uses `status`
    to decide whether to render the manifest or the "Unavailable" state.
    """

    plugin_id: str
    status: PluginStatus
    coming_soon: bool = False
    last_error: Optional[str] = None
    manifest: Optional[Manifest] = None


class PluginOptionItem(BaseModel):
    """One item in a dropdown resolved via `x-source`.

    Kept intentionally minimal — the frontend can pass extra keys through
    unchanged, but every option must have at least `value` and `label`.
    """

    value: str
    label: str


class PluginOptionsResponse(BaseModel):
    field: str
    options: list[PluginOptionItem]

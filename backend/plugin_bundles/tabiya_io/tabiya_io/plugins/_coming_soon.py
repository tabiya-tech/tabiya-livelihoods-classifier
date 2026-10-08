"""Shared stubs for coming-soon plugins.

A coming-soon plugin ships a real manifest (so the palette shows its name,
category, icon, and slots) but has no working implementation yet. Its manifest
sets `x-tabiya-coming-soon: true`; its invoke raises `UnavailableError` (503)
and its health reports `down`, so the orchestrator never treats it as usable.
"""

from __future__ import annotations

from tabiya_plugin_contracts import Context, Health, HealthStatus
from tabiya_plugin_contracts.adapters.http import UnavailableError


def make_coming_soon_invoke(plugin_label: str):
    """Return an invoke fn that always rejects with a coming-soon message."""

    async def invoke(input: object, config: dict, context: Context):
        raise UnavailableError(
            f"{plugin_label} is not implemented yet (coming soon).",
            detail={"coming_soon": True},
        )

    return invoke


async def coming_soon_health() -> Health:
    """Coming-soon plugins report `down` — they have no live implementation."""

    return Health(status=HealthStatus.DOWN, detail="coming_soon")

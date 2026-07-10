"""Job Scraper source plugin (COMING SOON).

Ships a real manifest so it appears in the palette, but has no working
implementation yet: invoke raises UnavailableError and health reports down.
When built, it will fetch a job posting from a URL and emit its text.
"""

from __future__ import annotations

from tabiya_plugin_contracts import Manifest, PluginCategory, Slot, SlotType

from .._coming_soon import coming_soon_health, make_coming_soon_invoke

MANIFEST = Manifest(
    plugin_id="tabiya.source.scraper.v1",
    name="Job Scraper",
    version="0.1.0",
    category=PluginCategory.SOURCE,
    summary="Fetches a job posting from a URL and feeds its text into the pipeline.",
    detail="coming soon",
    icon="globe",
    input_slot=Slot(type=SlotType.NONE, cardinality="none"),
    output_slot=Slot(type=SlotType.RAW_TEXT),
    config_schema={
        "type": "object",
        "properties": {
            "scrape_url": {
                "type": "string",
                "title": "URL",
                "description": "The job-posting URL to scrape.",
            }
        },
        "required": ["scrape_url"],
        "additionalProperties": False,
    },
    timeout_ms=15_000,
    **{"x-tabiya-coming-soon": True},
)

invoke = make_coming_soon_invoke("Job Scraper")
health = coming_soon_health

__all__ = ["MANIFEST", "invoke", "health"]

"""Installed plugins for the Tabiya-IO bundle.

Populated in 11.1b: text_input (Source), results (Sink), job_scraper.
For now the bundle boots empty so 11.1a can exercise the routing
scaffolding in isolation.
"""

from tabiya_plugin_contracts import Manifest


INSTALLED_PLUGINS: list[tuple[Manifest, object, object | None]] = []

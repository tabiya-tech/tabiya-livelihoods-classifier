"""Installed plugins for the Tabiya-Core bundle.

Populated in 11.1b: NER and NEL. For now the bundle boots empty so 11.1a
can exercise the routing scaffolding in isolation.
"""

from tabiya_plugin_contracts import Manifest
from tabiya_plugin_contracts.adapters.http import make_http_adapter
from fastapi import APIRouter


# Each installed plugin gets a tuple: (manifest, invoke_fn, optional health_fn).
# Populated by future subtasks. The `main.py` dispatcher iterates over
# INSTALLED_PLUGINS to mount every plugin's router.
INSTALLED_PLUGINS: list[tuple[Manifest, object, object | None]] = []

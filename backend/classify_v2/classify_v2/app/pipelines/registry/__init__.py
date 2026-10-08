"""Plugin registry — resolves catalog entries, fetches manifests, caches them.

Public surface:

  * `CatalogEntry`, `ResolvedPlugin`, `PluginStatus` — typed value objects.
  * `PluginRegistry` — the runtime cache. Exposes get/list/refresh.
  * Errors: `PluginUnreachableError`, `PluginManifestInvalidError`.
"""

from .errors import PluginManifestInvalidError, PluginUnreachableError
from .registry import DEFAULT_CATALOG_PATH, PluginRegistry, load_catalog
from .types import CatalogEntry, PluginStatus, ResolvedPlugin

__all__ = [
    "CatalogEntry",
    "DEFAULT_CATALOG_PATH",
    "PluginManifestInvalidError",
    "PluginRegistry",
    "PluginStatus",
    "PluginUnreachableError",
    "ResolvedPlugin",
    "load_catalog",
]

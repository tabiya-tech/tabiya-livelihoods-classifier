"""Executor-side errors.

These are distinct from the plugin adapter's `ErrorEnvelope` — they're the
executor's own view of a failed stage, carrying the stage index + plugin id
so the caller can render "stage 2 (tabiya.nel.v1) is unreachable" without
having to parse the plugin's raw error body.
"""

from __future__ import annotations

from typing import Any, Optional


class ExecutorError(Exception):
    """Base for executor errors."""

    def __init__(
        self,
        message: str,
        *,
        stage_index: int,
        plugin_id: str,
        detail: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.stage_index = stage_index
        self.plugin_id = plugin_id
        self.detail = detail


class PluginInvocationError(ExecutorError):
    """A plugin returned a non-2xx / malformed response.

    Mapped to HTTP 502 by the route: the classify caller's request was
    fine, the plugin is the failing party.
    """


class PluginTimeoutError(ExecutorError):
    """A plugin didn't respond within its manifest-declared timeout.

    Mapped to HTTP 504. Distinct from PluginInvocationError so ops
    dashboards can graph them separately.
    """


class PluginUpstreamUnavailableError(ExecutorError):
    """A plugin returned UPSTREAM_UNAVAILABLE (e.g. embeddings cache not ready).

    Mapped to HTTP 503 — preserves the pre-plugin classify behaviour where
    "cache not ready" was surfaced as 503, so callers with retry loops
    already handle this.
    """

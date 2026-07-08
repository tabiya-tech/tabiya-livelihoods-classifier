"""Pipeline executor.

Public surface:
  * `PipelineExecutor` — linear stage walker used by /v2/classify.
  * `ExecutorResult` — the shape it hands back to the classify route.
  * `PluginInvocationError` — non-2xx from a plugin's /invoke endpoint.
"""

from .errors import PluginInvocationError, PluginTimeoutError, PluginUpstreamUnavailableError
from .executor import ExecutorResult, PipelineExecutor, StageOutcome

__all__ = [
    "ExecutorResult",
    "PipelineExecutor",
    "PluginInvocationError",
    "PluginTimeoutError",
    "PluginUpstreamUnavailableError",
    "StageOutcome",
]

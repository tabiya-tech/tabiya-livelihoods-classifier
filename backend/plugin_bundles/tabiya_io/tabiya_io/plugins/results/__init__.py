"""Results sink plugin — LinkedEntities → NoneSlot.

The sink is intentionally minimal in v1: it consumes the linked entities
without further processing. The executor is responsible for exposing the
last-observed LinkedEntities payload to the classify_v2 caller (via
metadata / accumulator), so the sink itself only needs to accept the
input and return the `None` sentinel.

Future variants (email-a-summary, write-to-file, send-to-webhook) each
land as their own Sink plugin; this one is the default UI-facing sink.
"""

from .core import ResultsConfig, invoke
from .manifest import MANIFEST

__all__ = ["MANIFEST", "ResultsConfig", "invoke"]

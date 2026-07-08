"""Public surface of the plugin contracts package.

Everything a plugin author or orchestrator needs is re-exported here so
`from tabiya_plugin_contracts import Manifest, RawText, ...` is the single
import site.
"""

from .health import Health, HealthStatus
from .invoke import Context, ErrorCode, ErrorEnvelope, InvokeRequest, InvokeResponse
from .manifest import Manifest, PluginCategory
from .slots import (
    SLOT_MODEL_BY_TYPE,
    Entities,
    Entity,
    EntitySpan,
    LinkedEntities,
    LinkedEntity,
    Match,
    NoneSlot,
    RawText,
    RawTextStream,
    RawTextStreamItem,
    Slot,
    SlotType,
)
from .version import CONTRACT_VERSION

__all__ = [
    "CONTRACT_VERSION",
    "Context",
    "Entities",
    "Entity",
    "EntitySpan",
    "ErrorCode",
    "ErrorEnvelope",
    "Health",
    "HealthStatus",
    "InvokeRequest",
    "InvokeResponse",
    "LinkedEntities",
    "LinkedEntity",
    "Manifest",
    "Match",
    "NoneSlot",
    "PluginCategory",
    "RawText",
    "RawTextStream",
    "RawTextStreamItem",
    "SLOT_MODEL_BY_TYPE",
    "Slot",
    "SlotType",
]

"""NEL plugin — Entities → LinkedEntities."""

from .core import IEntityLinker, NelConfig, invoke
from .manifest import MANIFEST

__all__ = ["MANIFEST", "NelConfig", "IEntityLinker", "invoke"]

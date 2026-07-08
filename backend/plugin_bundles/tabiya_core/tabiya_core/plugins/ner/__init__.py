"""NER plugin — RawText → Entities.

Exports MANIFEST, NerConfig, invoke, IEntityExtractor so the bundle's
INSTALLED_PLUGINS registration is a single import site.
"""

from .core import IEntityExtractor, NerConfig, invoke
from .manifest import MANIFEST

__all__ = ["MANIFEST", "NerConfig", "IEntityExtractor", "invoke"]

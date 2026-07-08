"""Text-input source plugin — NoneSlot → RawText.

The source's `config` carries the payload the executor should push into
the pipeline. Two shapes are recognised:

  * `{ "text": "…" }` — single-body prose.
  * `{ "title": "…", "description": "…" }` — the shape most job ads land in.
    The two are joined with a blank line so downstream NER sees them as
    distinct paragraphs.

Anything else is a config error.
"""

from .core import TextInputConfig, invoke
from .manifest import MANIFEST

__all__ = ["MANIFEST", "TextInputConfig", "invoke"]

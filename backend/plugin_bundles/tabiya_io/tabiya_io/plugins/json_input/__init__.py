"""JSON-input source plugin — NoneSlot → RawText.

The source's `config` carries a JSON object (as a string) plus the name of
the field to read body text from:

  * `{ "json": "{\\"description\\": \\"Head chef wanted\\"}" }` → reads `text`
    (the default field), which is absent here, so it errors.
  * `{ "json": "{\\"description\\": \\"…\\"}", "text_field": "description" }`
    → reads the `description` field and pushes it downstream as RawText.

Invalid JSON, a non-object payload, a missing/empty/non-string target field,
or an empty payload are all config errors.
"""

from .core import JsonInputConfig, invoke
from .manifest import MANIFEST

__all__ = ["MANIFEST", "JsonInputConfig", "invoke"]

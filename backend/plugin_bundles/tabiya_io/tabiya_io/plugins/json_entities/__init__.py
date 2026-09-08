"""JSON-entities source plugin — NoneSlot → Entities.

Feeds a JSON array of pre-extracted entities straight into the pipeline as
`Entities`, so it can connect directly to the NEL stage and skip NER. Use
this to link a list you already have (e.g. occupations from a database) to
the taxonomy:

    json_entities → nel → results

Config carries the array as a string plus an optional `source_text`:

    { "json": "[{\\"surface_form\\": \\"head chef\\", \\"entity_type\\": \\"occupation\\"}]" }

Each item needs `surface_form` and `entity_type`; `span` is optional and
defaults to a zero span (pre-extracted entities have no offsets).
"""

from .core import JsonEntitiesConfig, invoke
from .manifest import MANIFEST

__all__ = ["MANIFEST", "JsonEntitiesConfig", "invoke"]

"""Classifications Mongo repository.

Public surface:
  * `ClassificationRecord` — persisted shape.
  * `ClassificationRepository`, `IClassificationRepository` — read/write.
"""

from ._types import ClassificationRecord
from .repository import (
    CLASSIFICATIONS_COLLECTION,
    CREATED_AT_INDEX_NAME,
    USER_INDEX_NAME,
    ClassificationRepository,
    IClassificationRepository,
)

__all__ = [
    "CLASSIFICATIONS_COLLECTION",
    "CREATED_AT_INDEX_NAME",
    "ClassificationRecord",
    "ClassificationRepository",
    "IClassificationRepository",
    "USER_INDEX_NAME",
]

"""
Shared helpers for building classifier input text and computing content hashes.
Used by the classify server, change-stream worker, and repository layer.
"""

import hashlib
from typing import Optional


def build_input_text(
    doc: dict,
    *,
    allow_text_field: bool = False,
) -> Optional[str]:
    """Build classifier input from a job dict.

    When *allow_text_field* is True, a non-empty ``text`` key is used as-is
    (the classify API accepts this). Workers that only have title+description
    should leave it False. Returns None when no usable text can be assembled.
    """
    if allow_text_field:
        text = doc.get("text")
        if text and isinstance(text, str) and text.strip():
            return text.strip()

    title = (doc.get("title") or "").strip()
    description = (doc.get("description") or "").strip()

    if title and description:
        return f"{title}.\n{description}"
    return description or title or None


def compute_hash(text: str) -> str:
    """SHA-256 hex digest of *text*, used for classification dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

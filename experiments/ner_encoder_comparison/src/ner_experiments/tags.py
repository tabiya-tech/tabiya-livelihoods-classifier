"""The tag scheme the models learn, and helpers to turn tag sequences into phrases (spans).

Every word gets one tag: ``O`` (outside any phrase), ``B-<type>`` (a phrase of that type
starts here) or ``I-<type>`` (the phrase continues).
"""

from enum import Enum


class EntityType(str, Enum):
    """The three phrase types the classifier links to the taxonomy."""

    OCCUPATION = "Occupation"
    SKILL = "Skill"
    QUALIFICATION = "Qualification"


ENTITY_TYPE_NAMES: list[str] = [entity_type.value for entity_type in EntityType]
OUTSIDE_TAG = "O"

# The position of a tag in this list is the number the model uses for it internally.
TAG_NAMES: list[str] = [OUTSIDE_TAG] + [
    f"{prefix}-{entity_type_name}"
    for entity_type_name in ENTITY_TYPE_NAMES
    for prefix in ("B", "I")
]
TAG_NAME_TO_ID: dict[str, int] = {tag_name: tag_id for tag_id, tag_name in enumerate(TAG_NAMES)}
TAG_ID_TO_NAME: dict[int, str] = dict(enumerate(TAG_NAMES))


def entity_type_of(tag: str) -> str | None:
    """``"B-Skill"`` -> ``"Skill"``; ``"O"`` -> ``None``."""
    if tag == OUTSIDE_TAG or "-" not in tag:
        return None
    return tag.split("-", 1)[1]


def repair_bio(word_tags: list[str]) -> list[str]:
    """Make a tag sequence valid: an ``I-`` tag that doesn't continue a phrase of the same
    type becomes a ``B-`` tag, starting a new phrase."""
    repaired_tags: list[str] = []
    previous_tag = OUTSIDE_TAG
    for tag in word_tags:
        tag_type = entity_type_of(tag)
        continues_previous_phrase = tag_type is not None and entity_type_of(previous_tag) == tag_type
        if tag.startswith("I-") and not continues_previous_phrase:
            tag = f"B-{tag_type}"
        repaired_tags.append(tag)
        previous_tag = tag
    return repaired_tags


def extract_spans(word_tags: list[str]) -> list[tuple[str, int, int]]:
    """Phrases in a (repaired) tag sequence as ``(entity_type, start_word, end_word)``,
    with ``end_word`` exclusive. A ``B-`` tag always starts a new phrase."""
    spans: list[tuple[str, int, int]] = []
    current_type: str | None = None
    current_start = 0
    for word_index, tag in enumerate(word_tags):
        tag_type = entity_type_of(tag)
        starts_new_phrase = tag.startswith("B-") or (tag_type is not None and tag_type != current_type)
        if current_type is not None and (tag_type is None or starts_new_phrase):
            spans.append((current_type, current_start, word_index))
            current_type = None
        if tag_type is not None and current_type is None:
            current_type = tag_type
            current_start = word_index
    if current_type is not None:
        spans.append((current_type, current_start, len(word_tags)))
    return spans

"""BIO tag post-processing utilities extracted from the original EntityLinker."""

from typing import List


def fix_bio_tags(tags: List[str]) -> List[str]:
    """Fix common BIO tagging errors.

    - B, O, I → B, I, I (gap fill)
    - O, I, O → O, O, O (isolated I removal)
    """
    fixed_tags = list(tags)
    for i in range(len(tags) - 2):
        if tags[i].startswith("B-") and tags[i + 1] == "O" and tags[i + 2].startswith("I-"):
            fixed_tags[i + 1] = tags[i + 2]
        if tags[i] == "O" and tags[i + 1].startswith("I-") and tags[i + 2] == "O":
            fixed_tags[i + 1] = "O"
    if len(tags) >= 2 and tags[-2] == "O" and tags[-1].startswith("I-"):
        fixed_tags[-1] = "O"
    return fixed_tags


def extract_entities(tokens: list, tags: list) -> List[dict]:
    """Convert parallel token/tag lists to entity dicts with 'type' and 'tokens' keys.

    A ``B-`` tag always starts a new entity, even when the previous token belongs to an
    entity of the same type, so two adjacent spans stay two spans. An ``I-`` tag continues
    the current entity when the type matches, and starts a new one otherwise.
    """
    result = []
    current_entity = None
    for token, tag in zip(tokens, tags):
        if tag == "O" or "-" not in tag:
            if current_entity:
                result.append(current_entity)
                current_entity = None
            continue

        tag_prefix, tag_label = tag.split("-", 1)
        continues_current_entity = (
            tag_prefix == "I"
            and current_entity is not None
            and current_entity["type"] == tag_label
        )
        if continues_current_entity:
            current_entity["tokens"].append(token)
        else:
            if current_entity:
                result.append(current_entity)
            current_entity = {"type": tag_label, "tokens": [token]}
    if current_entity:
        result.append(current_entity)
    return [item for item in result if len(item["tokens"]) != 0]

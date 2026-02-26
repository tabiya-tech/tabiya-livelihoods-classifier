"""
Shared NER post-processing helpers used by both inference/ner.py
and inference/linker.py.
"""

from typing import List, Tuple


def fix_bio_tags(tags: List[str]) -> List[str]:
    """Fix common BIO tagging errors: fill B-O-I gaps, remove orphan I-tags."""
    if len(tags) < 2:
        return list(tags)

    fixed = list(tags)

    for i in range(len(tags) - 2):
        if (
            tags[i].startswith("B-")
            and tags[i + 1] == "O"
            and tags[i + 2].startswith("I-")
        ):
            fixed[i + 1] = tags[i + 2]
        if (
            tags[i] == "O"
            and tags[i + 1].startswith("I-")
            and tags[i + 2] == "O"
        ):
            fixed[i + 1] = "O"

    if tags[-2] == "O" and tags[-1].startswith("I-"):
        fixed[-1] = "O"

    return fixed


def extract_entities(tokens: list, tags: list) -> List[dict]:
    """Convert parallel token-ID and BIO tag lists into entity spans."""
    result = []
    current_entity = None

    for token, tag in zip(tokens, tags):
        tag_type, tag_label = tag.split("-") if "-" in tag else ("O", tag)

        if tag_type != "O":
            if current_entity and current_entity["type"] == tag_label:
                current_entity["tokens"].append(token)
            else:
                if current_entity:
                    result.append(current_entity)
                current_entity = {"type": tag_label, "tokens": [token]}
        else:
            if current_entity:
                result.append(current_entity)
                current_entity = None

    if current_entity:
        result.append(current_entity)

    return [item for item in result if len(item["tokens"]) != 0]


def remove_special_tokens(
    input_ids: list,
    bio_tags: List[str],
    special_ids: set,
) -> Tuple[list, List[str]]:
    """Filter out special tokens (CLS, SEP, PAD) and their corresponding BIO tags."""
    filtered_ids = []
    filtered_tags = []

    for id_, tag in zip(input_ids, bio_tags):
        if id_ not in special_ids:
            filtered_ids.append(id_)
            filtered_tags.append(tag)

    return filtered_ids, filtered_tags

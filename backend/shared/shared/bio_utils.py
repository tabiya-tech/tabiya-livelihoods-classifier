"""BIO tag post-processing utilities extracted from the original EntityLinker."""

from typing import List, Tuple


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


def remove_special_tokens_and_tags(
    input_ids: List[int], bio_tags: List[str], tokenizer
) -> Tuple[List[int], List[str]]:
    """Filter out special token IDs and their corresponding BIO tags."""
    special_tokens_ids = tokenizer.all_special_ids
    filtered_ids = []
    filtered_tags = []
    for id_, tag in zip(input_ids, bio_tags):
        if id_ not in special_tokens_ids:
            filtered_ids.append(id_)
            filtered_tags.append(tag)
    return filtered_ids, filtered_tags


def extract_entities(tokens: list, tags: list) -> List[dict]:
    """Convert parallel token/tag lists to entity dicts with 'type' and 'tokens' keys."""
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

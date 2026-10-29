"""BIO tag post-processing utilities extracted from the original EntityLinker."""

from typing import List, Optional, Tuple


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


def spread_first_piece_tags(tags: List[str], word_ids: List[Optional[int]]) -> List[str]:
    """Give every piece of a word the tag predicted for that word's first piece.

    Token-classification models are trained with a label on the first piece (subword
    token) of each word only; the predictions for the other pieces were never trained and
    are noise. This keeps each word's first-piece tag and makes the word's other pieces
    continue it (``I-<type>``, or ``O`` when the word is outside every entity), so a word
    is never split between an entity and the text around it.

    ``word_ids`` is the tokenizer's piece-to-word map (``None`` for special tokens, whose
    tags are left unchanged).
    """
    spread_tags = list(tags)
    first_piece_tag_by_word = {}
    for piece_index, word_index in enumerate(word_ids):
        if word_index is None:
            continue
        if word_index not in first_piece_tag_by_word:
            first_piece_tag_by_word[word_index] = tags[piece_index]
            continue
        first_piece_tag = first_piece_tag_by_word[word_index]
        if first_piece_tag == "O" or "-" not in first_piece_tag:
            spread_tags[piece_index] = "O"
        else:
            spread_tags[piece_index] = "I-" + first_piece_tag.split("-", 1)[1]
    return spread_tags


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

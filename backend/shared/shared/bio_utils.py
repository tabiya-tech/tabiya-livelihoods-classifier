"""BIO tag post-processing utilities extracted from the original EntityLinker."""

import math
from typing import List, Sequence


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


def _continues(previous_tag: str, tag: str) -> bool:
    """Whether ``tag`` may follow ``previous_tag``: an I-X only continues a B-X or I-X."""
    if not tag.startswith("I-"):
        return True
    return previous_tag[2:] == tag[2:] and previous_tag[:2] in ("B-", "I-")


def best_valid_bio_tags(log_probs: Sequence[Sequence[float]], labels: Sequence[str]) -> List[str]:
    """The most likely tag sequence that is valid BIO (Viterbi over per-word log-probabilities).

    Taking each word's most likely tag on its own can give an I-X with no B-X before it,
    and patching that afterwards turns a low-confidence guess into a fragment
    ("plataforma/O de/I aprendizaje/I" -> "de aprendizaje"). Here the whole sentence is
    decided at once, so the answer is a whole phrase or none: never an orphan I-.
    """
    if not log_probs:
        return []
    allowed_first = [not label.startswith("I-") for label in labels]
    best = [score if allowed_first[i] else -math.inf for i, score in enumerate(log_probs[0])]
    back_pointers: List[List[int]] = []
    for word_scores in log_probs[1:]:
        pointers, scores = [], []
        for current, label in enumerate(labels):
            candidates = [
                (best[previous], previous)
                for previous, previous_label in enumerate(labels)
                if _continues(previous_label, label)
            ]
            score, previous = max(candidates)
            scores.append(score + word_scores[current])
            pointers.append(previous)
        best, back_pointers = scores, back_pointers + [pointers]
    position = max(range(len(labels)), key=lambda i: best[i])
    path = [position]
    for pointers in reversed(back_pointers):
        position = pointers[position]
        path.append(position)
    return [labels[i] for i in reversed(path)]


CLOSING_BRACKETS = {")": "(", "]": "[", "}": "{"}
OPENING_BRACKETS = {opening: closing for closing, opening in CLOSING_BRACKETS.items()}


def drop_edge_punctuation(words: Sequence[str], tags: Sequence[str]) -> List[str]:
    """Keep punctuation out of the first and last word of every phrase.

    "gestión del portal docente (" becomes "gestión del portal docente"; a phrase that is
    only punctuation disappears.
    """
    def is_punctuation(word: str) -> bool:
        return not any(character.isalnum() for character in word)

    def closes_a_bracket_inside(span_words: Sequence[str]) -> bool:
        """The last word closes a bracket opened inside the phrase: "(sincrónicas )" keeps its ")"."""
        text = "".join(span_words)
        return text[-1:] in CLOSING_BRACKETS and text.count(CLOSING_BRACKETS[text[-1]]) >= text.count(text[-1])

    def opens_a_bracket_inside(span_words: Sequence[str]) -> bool:
        text = "".join(span_words)
        return text[:1] in OPENING_BRACKETS and text.count(OPENING_BRACKETS[text[0]]) >= text.count(text[0])

    fixed = list(tags)
    index = 0
    while index < len(fixed):
        if not fixed[index].startswith("B-"):
            index += 1
            continue
        entity_type, end = fixed[index][2:], index + 1
        while end < len(fixed) and fixed[end] == f"I-{entity_type}":
            end += 1
        start, stop = index, end
        while start < stop and is_punctuation(words[start]) and not opens_a_bracket_inside(words[start:stop]):
            start += 1
        while stop > start and is_punctuation(words[stop - 1]) and not closes_a_bracket_inside(words[start:stop]):
            stop -= 1
        for position in range(index, end):
            fixed[position] = "O"
        for position in range(start, stop):
            fixed[position] = ("B-" if position == start else "I-") + entity_type
        index = end
    return fixed

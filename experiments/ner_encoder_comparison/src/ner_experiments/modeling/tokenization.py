"""Splitting words into the model's pieces (subword tokens) and lining the tags up.

Each model ships its own tokenizer, loaded from Hugging Face. We pass it the words as
already split by the corpus (``is_split_into_words``) and use ``word_ids()`` to know which
word each piece came from. Only the first piece of each word carries the word's tag; the
other pieces are ignored when training (``IGNORED_IN_LOSS``).
"""

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ner_experiments.tags import OUTSIDE_TAG, TAG_ID_TO_NAME, TAG_NAME_TO_ID

IGNORED_IN_LOSS = -100


def load_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
    """Load the model's fast tokenizer. RoBERTa-style tokenizers refuse pre-split words
    unless created with ``add_prefix_space=True``, so retry with it when needed."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError(f"{model_name} has no fast tokenizer; word alignment needs one")
    try:
        tokenizer(["prueba", "de", "palabras"], is_split_into_words=True)
    except (ValueError, AssertionError, TypeError):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=True)
    return tokenizer


def pieces_per_word(tokenizer: PreTrainedTokenizerFast, words: list[str]) -> list[int]:
    """How many pieces the tokenizer splits each word into."""
    # verbose=False: this only counts pieces, so the "longer than the maximum" warning does
    # not apply (long sentences are cut into chunks afterwards).
    encoded = tokenizer(words, is_split_into_words=True, add_special_tokens=False, verbose=False)
    piece_counts = [0] * len(words)
    for word_index in encoded.word_ids():
        if word_index is not None:
            piece_counts[word_index] += 1
    return piece_counts


def chunk_word_ranges(tokenizer: PreTrainedTokenizerFast, words: list[str], max_pieces: int) -> list[tuple[int, int]]:
    """Cut a sentence into consecutive word ranges that each fit in ``max_pieces`` pieces
    (special tokens included). Almost every sentence fits whole and returns one range."""
    piece_budget = max_pieces - tokenizer.num_special_tokens_to_add(pair=False)
    word_ranges: list[tuple[int, int]] = []
    chunk_start = 0
    pieces_in_chunk = 0
    for word_index, piece_count in enumerate(pieces_per_word(tokenizer, words)):
        if pieces_in_chunk + piece_count > piece_budget and word_index > chunk_start:
            word_ranges.append((chunk_start, word_index))
            chunk_start = word_index
            pieces_in_chunk = 0
        pieces_in_chunk += piece_count
    word_ranges.append((chunk_start, len(words)))
    return word_ranges


def encode_with_first_piece_labels(
    tokenizer: PreTrainedTokenizerFast, words: list[str], word_tags: list[str], max_pieces: int
) -> dict:
    """Model inputs for one chunk of words, with each word's tag on its first piece."""
    encoded = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_pieces)
    piece_labels: list[int] = []
    previous_word_index = None
    for word_index in encoded.word_ids():
        is_special_piece = word_index is None
        continues_previous_word = word_index == previous_word_index
        if is_special_piece or continues_previous_word:
            piece_labels.append(IGNORED_IN_LOSS)
        else:
            piece_labels.append(TAG_NAME_TO_ID[word_tags[word_index]])
        previous_word_index = word_index
    encoded["labels"] = piece_labels
    return encoded


def first_piece_labels_to_word_tags(piece_labels: list[int], word_count: int) -> list[str]:
    """Undo ``encode_with_first_piece_labels``: one tag per word from the labelled pieces."""
    word_tags = [label_id for label_id in piece_labels if label_id != IGNORED_IN_LOSS]
    tags = [TAG_ID_TO_NAME[label_id] for label_id in word_tags]
    return tags + [OUTSIDE_TAG] * (word_count - len(tags))

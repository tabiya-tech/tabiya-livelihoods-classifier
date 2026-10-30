"""Predict one tag per word, from each word's first piece (the piece that carried the tag
in training)."""

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from ner_experiments.modeling.tokenization import chunk_word_ranges
from ner_experiments.schemas import PredictedSentence, TaggedSentence
from ner_experiments.tags import OUTSIDE_TAG, TAG_ID_TO_NAME, repair_bio


def predict_sentences(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    sentences: list[TaggedSentence],
    max_pieces: int,
    device: str,
    chunks_per_batch: int = 32,
    description: str = "Predicting",
) -> list[PredictedSentence]:
    """Predicted word tags for every sentence. Long sentences are predicted in chunks and
    stitched back together."""
    model.eval()
    model.to(device)

    # Every chunk remembers which sentence and which words it covers.
    chunk_owners: list[tuple[int, int, int]] = []
    for sentence_index, sentence in enumerate(sentences):
        for range_start, range_end in chunk_word_ranges(tokenizer, sentence.tokens, max_pieces):
            chunk_owners.append((sentence_index, range_start, range_end))

    predicted_tags_per_sentence = [[OUTSIDE_TAG] * len(sentence.tokens) for sentence in sentences]
    batch_starts = range(0, len(chunk_owners), chunks_per_batch)
    for batch_start in tqdm(batch_starts, desc=description, unit="batch"):
        batch_owners = chunk_owners[batch_start:batch_start + chunks_per_batch]
        batch_words = [sentences[owner[0]].tokens[owner[1]:owner[2]] for owner in batch_owners]
        encoded_batch = tokenizer(
            batch_words, is_split_into_words=True, truncation=True, max_length=max_pieces,
            padding=True, return_tensors="pt",
        )
        model_inputs = {input_name: input_tensor.to(device) for input_name, input_tensor in encoded_batch.items()}
        with torch.no_grad():
            # One score per tag for every piece of every chunk in the batch.
            scores_per_piece = model(**model_inputs).logits
        best_tag_id_per_piece = scores_per_piece.argmax(dim=-1).cpu().tolist()

        for position_in_batch, (sentence_index, range_start, _range_end) in enumerate(batch_owners):
            words_already_tagged: set[int] = set()
            for piece_position, word_index in enumerate(encoded_batch.word_ids(position_in_batch)):
                is_first_piece_of_word = word_index is not None and word_index not in words_already_tagged
                if is_first_piece_of_word:
                    tag_id = best_tag_id_per_piece[position_in_batch][piece_position]
                    predicted_tags_per_sentence[sentence_index][range_start + word_index] = TAG_ID_TO_NAME[tag_id]
                    words_already_tagged.add(word_index)

    return [
        PredictedSentence(**sentence.model_dump(), predicted_tags=repair_bio(predicted_tags))
        for sentence, predicted_tags in zip(sentences, predicted_tags_per_sentence)
    ]

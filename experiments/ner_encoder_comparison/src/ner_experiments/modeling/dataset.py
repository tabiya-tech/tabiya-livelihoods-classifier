"""Turn tagged sentences into the encoded examples the trainer consumes."""

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from ner_experiments.modeling.tokenization import chunk_word_ranges, encode_with_first_piece_labels
from ner_experiments.schemas import TaggedSentence


class EncodedSentenceDataset(torch.utils.data.Dataset):
    """Encoded sentences (long ones split into chunks), each piece-labelled for training."""

    def __init__(
        self,
        sentences: list[TaggedSentence],
        tokenizer: PreTrainedTokenizerFast,
        max_pieces: int,
        description: str,
    ) -> None:
        self.encoded_chunks: list[dict] = []
        self.chunked_sentence_count = 0
        for sentence in tqdm(sentences, desc=description, unit="sentence"):
            word_ranges = chunk_word_ranges(tokenizer, sentence.tokens, max_pieces)
            if len(word_ranges) > 1:
                self.chunked_sentence_count += 1
            for range_start, range_end in word_ranges:
                self.encoded_chunks.append(
                    encode_with_first_piece_labels(
                        tokenizer,
                        sentence.tokens[range_start:range_end],
                        sentence.tags[range_start:range_end],
                        max_pieces,
                    )
                )

    def __len__(self) -> int:
        return len(self.encoded_chunks)

    def __getitem__(self, chunk_index: int) -> dict:
        return self.encoded_chunks[chunk_index]

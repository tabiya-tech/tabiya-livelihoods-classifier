"""Sanity check before training: does each model's tokenizer line the tags up correctly?

For each model, encode sentences, read the tags back from the labelled first pieces, and
check they match the original tags word for word.

    python -m ner_experiments.cli.check_tokenizers
    python -m ner_experiments.cli.check_tokenizers --models BSC-LT/mRoBERTa --sentences 200
"""

import argparse
import sys

from tqdm.auto import tqdm

from ner_experiments.config import DataPaths
from ner_experiments.corpus.loading import read_sentences
from ner_experiments.modeling.tokenization import (
    chunk_word_ranges,
    encode_with_first_piece_labels,
    first_piece_labels_to_word_tags,
    load_tokenizer,
)

DEFAULT_MODELS = [
    "FacebookAI/xlm-roberta-base",
    "BSC-LT/mRoBERTa",
    "jhu-clsp/mmBERT-base",
    "BSC-LT/MrBERT-es",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--sentences", type=int, default=50)
    parser.add_argument("--max-pieces", type=int, default=256)
    arguments = parser.parse_args()

    sentences = read_sentences(DataPaths().processed_dir, "train")
    tagged_sentences = [sentence for sentence in sentences if any(tag != "O" for tag in sentence.tags)]
    longest_sentences = sorted(sentences, key=lambda sentence: -len(sentence.tokens))[:10]
    sample = tagged_sentences[: arguments.sentences] + longest_sentences

    all_passed = True
    for model_name in arguments.models:
        tokenizer = load_tokenizer(model_name)
        mismatches = 0
        chunked = 0
        for sentence in tqdm(sample, desc=model_name, unit="sentence"):
            word_ranges = chunk_word_ranges(tokenizer, sentence.tokens, arguments.max_pieces)
            chunked += len(word_ranges) > 1
            recovered_tags: list[str] = []
            for range_start, range_end in word_ranges:
                encoded = encode_with_first_piece_labels(
                    tokenizer, sentence.tokens[range_start:range_end], sentence.tags[range_start:range_end],
                    arguments.max_pieces,
                )
                recovered_tags += first_piece_labels_to_word_tags(encoded["labels"], range_end - range_start)
            if recovered_tags != sentence.tags:
                mismatches += 1
        passed = mismatches == 0
        all_passed &= passed
        print(
            f"{'PASS' if passed else 'FAIL'}  {model_name}: {len(sample) - mismatches}/{len(sample)} sentences "
            f"round-trip exactly; {chunked} long sentence(s) needed splitting into chunks"
        )
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

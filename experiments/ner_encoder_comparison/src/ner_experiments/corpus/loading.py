"""Read and write the processed train / dev / test files (one JSON sentence per line)."""

import random
from collections import Counter
from pathlib import Path

from ner_experiments.schemas import SplitName, TaggedSentence


def processed_file(processed_dir: Path, split_name: SplitName) -> Path:
    return processed_dir / f"{split_name}.jsonl"


def write_sentences(path: Path, sentences: list[TaggedSentence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        for sentence in sentences:
            output_file.write(sentence.model_dump_json() + "\n")


def read_sentences(processed_dir: Path, split_name: SplitName) -> list[TaggedSentence]:
    path = processed_file(processed_dir, split_name)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m ner_experiments.cli.prepare_data` first")
    with open(path, encoding="utf-8") as input_file:
        return [TaggedSentence.model_validate_json(line) for line in input_file if line.strip()]


def keep_share_of_ads(sentences: list[TaggedSentence], share: float, seed: int) -> list[TaggedSentence]:
    """Keep a random share of the ads (whole ads, not sentences), for the learning curve."""
    if share >= 1:
        return sentences
    ad_ids = sorted({sentence.ad_id for sentence in sentences})
    kept_ad_ids = set(random.Random(seed).sample(ad_ids, max(1, round(len(ad_ids) * share))))
    return [sentence for sentence in sentences if sentence.ad_id in kept_ad_ids]


def count_phrases(sentences: list[TaggedSentence]) -> Counter:
    """How many phrases of each type the sentences contain."""
    phrase_counts: Counter = Counter()
    for sentence in sentences:
        phrase_counts.update(tag[2:] for tag in sentence.tags if tag.startswith("B-"))
    return phrase_counts

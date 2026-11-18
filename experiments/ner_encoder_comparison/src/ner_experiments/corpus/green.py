"""The English Green job-ad corpus, as Tabiya's English model was trained on it.

Source: tabiya/job_ner_dataset on Hugging Face (Green et al. 2022, CC BY 4.0): 9,634 train and 336
test sentences from UK job ads. Its tags are numbers; the order below was recovered by lining its
rows up with jjzha/green, the original release.

Green tags five types. Skill, Occupation and Qualification map onto ours; Experience ("3 years")
and Domain (the job's sector) have no equivalent and become 'O', as the pipeline drops them anyway.
The release merges Green's train and dev sets, so a fixed share of train is held out as dev.
"""

import random
from pathlib import Path

from ner_experiments.schemas import SplitName, TaggedSentence
from ner_experiments.tags import repair_bio

GREEN_DATASET = "tabiya/job_ner_dataset"
GREEN_FILES: dict[str, str] = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
GREEN_TAG_NAMES = [
    "O", "B-Skill", "B-Qualification", "I-Domain", "I-Experience", "I-Qualification",
    "B-Occupation", "B-Domain", "I-Occupation", "I-Skill", "B-Experience",
]
_KEPT_TYPES = {"Skill", "Occupation", "Qualification"}


def download_green(raw_dir: Path) -> dict[str, Path]:
    """Fetch (or reuse) the two parquet files under raw_dir."""
    from huggingface_hub import hf_hub_download

    return {
        split: Path(hf_hub_download(GREEN_DATASET, file_name, repo_type="dataset", local_dir=raw_dir))
        for split, file_name in GREEN_FILES.items()
    }


def convert_green_tag(tag_id: int) -> str:
    """Green tag number -> our tag: kept types as they are, Experience and Domain -> 'O'."""
    tag = GREEN_TAG_NAMES[tag_id]
    return tag if tag != "O" and tag[2:] in _KEPT_TYPES else "O"


def read_green(parquet_path: Path, source_split: str) -> list[TaggedSentence]:
    """Every sentence of one release split, converted to our tags. Each sentence is its own 'ad'."""
    import pyarrow.parquet as pq

    sentences = []
    for row in pq.read_table(parquet_path, columns=["id", "tokens", "ner_tags"]).to_pylist():
        if not row["tokens"]:
            continue
        sentences.append(
            TaggedSentence(
                ad_id=f"green/{source_split}/{row['id']}",
                sentence_index=0,
                tokens=list(row["tokens"]),
                tags=repair_bio([convert_green_tag(tag_id) for tag_id in row["ner_tags"]]),
            )
        )
    return sentences


def hold_out_dev(train: list[TaggedSentence], dev_share: float, seed: int) -> dict[SplitName, list[TaggedSentence]]:
    """Split the release's train into train and dev, the same way every time for a given seed."""
    shuffled = list(train)
    random.Random(seed).shuffle(shuffled)
    dev_count = round(len(shuffled) * dev_share)
    dev_ids = {sentence.ad_id for sentence in shuffled[:dev_count]}
    return {
        "train": [s for s in train if s.ad_id not in dev_ids],
        "dev": [s for s in train if s.ad_id in dev_ids],
    }

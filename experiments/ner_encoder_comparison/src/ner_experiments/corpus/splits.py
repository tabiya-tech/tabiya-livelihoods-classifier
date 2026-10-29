"""Train / dev / test split by whole ad, frozen as ID lists in ``splits/``.

Splitting by ad (not by sentence) keeps one ad's title, boilerplate and wording out of
both sides; splitting by folder first keeps the random vs degree-required mix the same in
every split (stratified split). The ID lists are committed to git, so every run, on every
machine, uses exactly the same split.
"""

import random
from pathlib import Path

from ner_experiments.schemas import SplitName

SPLIT_NAMES: tuple[SplitName, ...] = ("train", "dev", "test")


def make_split(
    ad_ids_by_folder: dict[str, list[str]],
    train_share: float,
    dev_share: float,
    seed: int,
) -> dict[SplitName, list[str]]:
    """Shuffle each folder's ads with a fixed seed and cut them into train / dev / test."""
    random_generator = random.Random(seed)
    split_ad_ids: dict[SplitName, list[str]] = {split_name: [] for split_name in SPLIT_NAMES}
    for folder in sorted(ad_ids_by_folder):
        folder_ad_ids = sorted(ad_ids_by_folder[folder])
        random_generator.shuffle(folder_ad_ids)
        train_end = round(len(folder_ad_ids) * train_share)
        dev_end = train_end + round(len(folder_ad_ids) * dev_share)
        split_ad_ids["train"] += folder_ad_ids[:train_end]
        split_ad_ids["dev"] += folder_ad_ids[train_end:dev_end]
        split_ad_ids["test"] += folder_ad_ids[dev_end:]
    return {split_name: sorted(ad_ids) for split_name, ad_ids in split_ad_ids.items()}


def split_file(splits_dir: Path, split_name: SplitName) -> Path:
    return splits_dir / f"{split_name}_ads.txt"


def splits_exist(splits_dir: Path) -> bool:
    return all(split_file(splits_dir, split_name).exists() for split_name in SPLIT_NAMES)


def write_split(splits_dir: Path, split_ad_ids: dict[SplitName, list[str]]) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, ad_ids in split_ad_ids.items():
        split_file(splits_dir, split_name).write_text("\n".join(ad_ids) + "\n", encoding="utf-8")


def read_split(splits_dir: Path) -> dict[SplitName, list[str]]:
    split_ad_ids: dict[SplitName, list[str]] = {}
    for split_name in SPLIT_NAMES:
        lines = split_file(splits_dir, split_name).read_text(encoding="utf-8").splitlines()
        split_ad_ids[split_name] = [line.strip() for line in lines if line.strip()]
    all_ad_ids = [ad_id for ad_ids in split_ad_ids.values() for ad_id in ad_ids]
    if len(all_ad_ids) != len(set(all_ad_ids)):
        raise ValueError(f"an ad appears in more than one split file in {splits_dir}")
    return split_ad_ids

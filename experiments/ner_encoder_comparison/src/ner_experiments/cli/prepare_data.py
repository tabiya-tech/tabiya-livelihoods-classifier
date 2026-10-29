"""Download the Peru corpus, convert its tags, and write train / dev / test files.

    python -m ner_experiments.cli.prepare_data [--config configs/prepare.yaml]

The split is read from ``splits/*.txt`` (committed to git). It is only created when those
files are missing, or when you pass ``--new-split`` on purpose.
"""

import argparse
import csv
from pathlib import Path

from ner_experiments.config import PrepareConfig, load_settings
from ner_experiments.corpus.loading import count_phrases, processed_file, write_sentences
from ner_experiments.corpus.peru_corpus import download_corpus, read_corpus
from ner_experiments.corpus.splits import SPLIT_NAMES, make_split, read_split, split_file, splits_exist, write_split
from ner_experiments.progress import StageReporter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="YAML file with PrepareConfig settings")
    parser.add_argument("--new-split", action="store_true", help="Make a new split even if splits/ exists")
    arguments = parser.parse_args()

    stages = StageReporter(total_stages=4)
    stages.start("Settings")
    settings = load_settings(PrepareConfig, arguments.config)
    print(settings.model_dump_json(indent=2))

    stages.start("Getting the Peru corpus")
    zip_path = download_corpus(settings.corpus_url, settings.paths.raw_dir)
    ads, conversion_report = read_corpus(zip_path, settings.task_description_spans, settings.move_qualification_levels)
    print(f"Read {len(ads)} ads, {sum(len(ad.sentences) for ad in ads)} sentences")
    if settings.move_qualification_levels:
        review_path = settings.paths.processed_dir / "moved_to_qualification.csv"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        with open(review_path, "w", newline="", encoding="utf-8") as review_file:
            writer = csv.writer(review_file)
            writer.writerow(["ad_id", "sentence_index", "phrase"])
            for moved in conversion_report.moved_to_qualification:
                writer.writerow([moved.ad_id, moved.sentence_index, moved.phrase])
        print(
            f"Moved {len(conversion_report.moved_to_qualification)} REQ phrases from Skill to Qualification; "
            f"review them in {review_path}"
        )

    stages.start("Train / dev / test split by whole ad")
    if splits_exist(settings.paths.splits_dir) and not arguments.new_split:
        split_ad_ids = read_split(settings.paths.splits_dir)
        print(f"Using the frozen split in {settings.paths.splits_dir}/")
    else:
        ad_ids_by_folder: dict[str, list[str]] = {}
        for ad in ads:
            ad_ids_by_folder.setdefault(ad.folder, []).append(ad.ad_id)
        split_ad_ids = make_split(ad_ids_by_folder, settings.train_share, settings.dev_share, settings.split_seed)
        write_split(settings.paths.splits_dir, split_ad_ids)
        print(f"Wrote a NEW split to {settings.paths.splits_dir}/; commit these files so every run uses them")

    known_ad_ids = {ad.ad_id for ad in ads}
    for split_name in SPLIT_NAMES:
        missing = [ad_id for ad_id in split_ad_ids[split_name] if ad_id not in known_ad_ids]
        if missing:
            raise ValueError(f"{split_file(settings.paths.splits_dir, split_name)} names unknown ads: {missing[:5]}")

    stages.start("Writing processed files")
    ads_by_id = {ad.ad_id: ad for ad in ads}
    print(f"{'split':<8}{'ads':>6}{'sentences':>11}{'words':>9}{'Occupation':>12}{'Skill':>8}{'Qualification':>15}")
    for split_name in SPLIT_NAMES:
        sentences = [sentence for ad_id in split_ad_ids[split_name] for sentence in ads_by_id[ad_id].sentences]
        write_sentences(processed_file(settings.paths.processed_dir, split_name), sentences)
        phrase_counts = count_phrases(sentences)
        print(
            f"{split_name:<8}{len(split_ad_ids[split_name]):>6}{len(sentences):>11}"
            f"{sum(len(sentence.tokens) for sentence in sentences):>9}{phrase_counts['Occupation']:>12}"
            f"{phrase_counts['Skill']:>8}{phrase_counts['Qualification']:>15}"
        )
    print(f"\nWrote {', '.join(str(processed_file(settings.paths.processed_dir, name)) for name in SPLIT_NAMES)}")
    stages.finish()


if __name__ == "__main__":
    main()

"""Prepare the English Green corpus for mixed English + Spanish training.

    python -m ner_experiments.cli.prepare_green

Downloads tabiya/job_ner_dataset (the data behind Tabiya's English model) to data/raw/green/,
maps its tags to ours (Experience and Domain become 'O') and writes
data/processed-green/{train,dev,test}.jsonl. A run uses it with `mix_with: [data/processed-green]`.
"""

import argparse
from pathlib import Path

from ner_experiments.config import EXPERIMENT_DIR, DataPaths
from ner_experiments.corpus.green import download_green, hold_out_dev, read_green
from ner_experiments.corpus.loading import count_phrases, processed_file, write_sentences
from ner_experiments.progress import StageReporter
from ner_experiments.tags import EntityType, extract_spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=EXPERIMENT_DIR / "data" / "processed-green")
    parser.add_argument("--dev-share", type=float, default=0.1, help="Share of the release's train held out as dev")
    parser.add_argument("--seed", type=int, default=13)
    arguments = parser.parse_args()

    stages = StageReporter(total_stages=3)
    stages.start("Downloading tabiya/job_ner_dataset")
    files = download_green(DataPaths().raw_dir / "green")

    stages.start("Converting tags")
    splits = hold_out_dev(read_green(files["train"], "train"), arguments.dev_share, arguments.seed)
    splits["test"] = read_green(files["test"], "test")

    stages.start("Writing")
    arguments.output.mkdir(parents=True, exist_ok=True)
    print(f"{'split':<6}{'sentences':>10}{'Occupation':>12}{'Skill':>8}{'Qualification':>15}{'words/skill':>13}")
    for split, sentences in splits.items():
        write_sentences(processed_file(arguments.output, split), sentences)
        phrases = count_phrases(sentences)
        skill_lengths = [end - start for s in sentences for kind, start, end in extract_spans(s.tags) if kind == EntityType.SKILL.value]
        mean_length = sum(skill_lengths) / len(skill_lengths) if skill_lengths else 0
        print(f"{split:<6}{len(sentences):>10}{phrases['Occupation']:>12}{phrases['Skill']:>8}{phrases['Qualification']:>15}{mean_length:>13.1f}")
    print(f"Written to {arguments.output}")
    stages.finish()


if __name__ == "__main__":
    main()

"""A table of every finished run, written as CSV (for spreadsheets) and Markdown (to read)."""

import csv
from pathlib import Path

from ner_experiments.schemas import RunResult, SplitScores
from ner_experiments.tags import EntityType

_COLUMNS = [
    "run_name", "model_name", "learning_rate", "seed", "train_fraction", "epochs_trained", "train_minutes",
    "dev_macro_f1", "dev_occupation_f1", "dev_skill_f1", "dev_qualification_f1", "dev_skill_words",
    "test_macro_f1",
]


def _row(result: RunResult) -> dict[str, str]:
    dev: SplitScores = result.dev

    def percent(value: float) -> str:
        return f"{100 * value:.1f}"

    return {
        "run_name": result.run_name,
        "model_name": result.model_name,
        "learning_rate": f"{result.learning_rate:g}",
        "seed": str(result.seed),
        "train_fraction": f"{result.train_fraction:g}",
        "epochs_trained": f"{result.epochs_trained:.0f}",
        "train_minutes": f"{result.train_minutes:.1f}",
        "dev_macro_f1": percent(dev.macro_strict_f1),
        "dev_occupation_f1": percent(dev.strict[EntityType.OCCUPATION].f1),
        "dev_skill_f1": percent(dev.strict[EntityType.SKILL].f1),
        "dev_qualification_f1": percent(dev.strict[EntityType.QUALIFICATION].f1),
        "dev_skill_words": f"{dev.skill_span_length.predicted_mean_words:.2f}",
        "test_macro_f1": percent(result.test.macro_strict_f1) if result.test else "",
    }


def write_summary(results: list[RunResult], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_row(result) for result in sorted(results, key=lambda result: (result.model_name, result.learning_rate, result.seed))]

    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_dir / "summary.md"
    header = "| " + " | ".join(_COLUMNS) + " |"
    divider = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    body = ["| " + " | ".join(row[column] for column in _COLUMNS) + " |" for row in rows]
    markdown_path.write_text("\n".join([header, divider, *body]) + "\n", encoding="utf-8")
    return csv_path, markdown_path

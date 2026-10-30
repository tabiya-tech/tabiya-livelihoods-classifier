"""The files every run writes into its own folder, ``runs/<run name>/``.

Each run only ever touches its own folder, so several runs can safely go at the same time.

    effective_config.yaml       settings actually used (YAML + environment overrides)
    result.json                 scores, history and timing (a RunResult)
    predictions_dev.jsonl       every dev sentence with real and predicted tags
    predictions_test.jsonl      the same for test (only written when evaluate_on_test is on)
    training_loss.json          training loss every 25 steps, for the charts
    charts/*.png                per-run charts
    model/                      the trained model (only when keep_model is on)
"""

import json
from pathlib import Path

import yaml

from ner_experiments.schemas import PredictedSentence, RunResult, SplitName

RESULT_FILE = "result.json"


def write_effective_config(run_dir: Path, config_values: dict) -> None:
    (run_dir / "effective_config.yaml").write_text(yaml.safe_dump(config_values, sort_keys=False), encoding="utf-8")


def write_result(run_dir: Path, result: RunResult) -> None:
    (run_dir / RESULT_FILE).write_text(result.model_dump_json(indent=2), encoding="utf-8")


def read_result(run_dir: Path) -> RunResult:
    return RunResult.model_validate_json((run_dir / RESULT_FILE).read_text(encoding="utf-8"))


def predictions_file(run_dir: Path, split: SplitName) -> Path:
    return run_dir / f"predictions_{split}.jsonl"


def write_predictions(run_dir: Path, split: SplitName, predictions: list[PredictedSentence]) -> None:
    with open(predictions_file(run_dir, split), "w", encoding="utf-8") as output_file:
        for sentence in predictions:
            output_file.write(sentence.model_dump_json() + "\n")


def read_predictions(run_dir: Path, split: SplitName) -> list[PredictedSentence]:
    path = predictions_file(run_dir, split)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing (was evaluate_on_test on for this run?)")
    with open(path, encoding="utf-8") as input_file:
        return [PredictedSentence.model_validate_json(line) for line in input_file if line.strip()]


def write_loss_curve(run_dir: Path, loss_curve: list[tuple[int, float]]) -> None:
    (run_dir / "training_loss.json").write_text(json.dumps(loss_curve), encoding="utf-8")


def read_loss_curve(run_dir: Path) -> list[tuple[int, float]]:
    path = run_dir / "training_loss.json"
    return [tuple(point) for point in json.loads(path.read_text(encoding="utf-8"))] if path.exists() else []


def find_finished_runs(runs_dir: Path) -> list[Path]:
    return sorted(path.parent for path in runs_dir.glob(f"*/{RESULT_FILE}"))


def is_final_run(result: RunResult) -> bool:
    """A final run: trained on all training ads with the test set scored."""
    return result.test is not None and result.train_fraction == 1

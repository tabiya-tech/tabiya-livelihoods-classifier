"""Are the gaps between models real or luck? Paired bootstrap over ads.

Groups finished runs by model (each model's repeat runs are averaged), then applies the
decision rule's significance check. By default only the final runs are used: full-data runs
trained with evaluate_on_test on. Learning-rate sweep runs are left out so each model is
judged at its chosen setting only.

    python -m ner_experiments.cli.compare --split test
    python -m ner_experiments.cli.compare --split dev --runs runs/a runs/b runs/c
"""

import argparse
from collections import defaultdict
from pathlib import Path

from ner_experiments.config import DataPaths
from ner_experiments.evaluation.bootstrap import paired_bootstrap
from ner_experiments.run_files import find_finished_runs, is_final_run, read_predictions, read_result
from ner_experiments.schemas import PredictedSentence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--runs", type=Path, nargs="*", help="Run folders to use instead of the final runs")
    parser.add_argument("--runs-dir", type=Path, default=DataPaths().runs_dir)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DataPaths().reports_dir / "latest")
    arguments = parser.parse_args()

    if arguments.runs:
        run_dirs = arguments.runs
    else:
        run_dirs = [run_dir for run_dir in find_finished_runs(arguments.runs_dir) if is_final_run(read_result(run_dir))]
    predictions_by_model: dict[str, list[list[PredictedSentence]]] = defaultdict(list)
    for run_dir in run_dirs:
        result = read_result(run_dir)
        if arguments.split == "test" and result.test is None:
            raise SystemExit(f"{run_dir} was trained with evaluate_on_test off, so it has no test predictions")
        predictions_by_model[result.model_name].append(read_predictions(run_dir, arguments.split))
        print(f"  {result.model_name:<35} {run_dir.name}")
    if len(predictions_by_model) < 2:
        raise SystemExit(
            "Need runs from at least two models. By default only final runs count (full data, "
            "evaluate_on_test on); pass --runs to pick run folders yourself."
        )

    comparison = paired_bootstrap(predictions_by_model, arguments.split, arguments.resamples)

    print(f"\n{arguments.split} set, {comparison.ads} ads, {comparison.resamples} resamples")
    print(f"{'model':<35}{'runs':>6}{'mean F1':>10}{'95% range':>18}")
    for model in comparison.models:
        print(
            f"{model.model_name:<35}{model.runs:>6}{100 * model.mean_macro_strict_f1:>10.1f}"
            f"{100 * model.interval_low:>9.1f} to {100 * model.interval_high:.1f}"
        )
    print("\nGaps between models (exact-match macro F1 points):")
    for gap in comparison.gaps:
        verdict = "TIED (too close to call)" if gap.tied else "real difference"
        print(
            f"  {gap.better_model} vs {gap.worse_model}: +{100 * gap.mean_gap:.1f} "
            f"(95% range {100 * gap.interval_low:+.1f} to {100 * gap.interval_high:+.1f}) -> {verdict}"
        )

    arguments.output.mkdir(parents=True, exist_ok=True)
    output_path = arguments.output / f"bootstrap_{arguments.split}.json"
    output_path.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()

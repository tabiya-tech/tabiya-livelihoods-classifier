"""Collect every finished run into a summary table and comparison charts.

    python -m ner_experiments.cli.report
    python -m ner_experiments.cli.report --runs-dir runs --output reports/latest --include-test
"""

import argparse
from pathlib import Path

from ner_experiments.config import DataPaths
from ner_experiments.reporting.charts import plot_learning_curve, plot_learning_rate_sweep, plot_model_comparison
from ner_experiments.reporting.summary import write_summary
from ner_experiments.run_files import find_finished_runs, is_final_run, read_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=DataPaths().runs_dir)
    parser.add_argument("--output", type=Path, default=DataPaths().reports_dir / "latest")
    parser.add_argument("--include-test", action="store_true", help="Also chart test scores (final runs only)")
    arguments = parser.parse_args()

    run_dirs = find_finished_runs(arguments.runs_dir)
    if not run_dirs:
        raise SystemExit(f"No finished runs in {arguments.runs_dir}")
    results = [read_result(run_dir) for run_dir in run_dirs]
    print(f"Found {len(results)} finished runs")

    csv_path, markdown_path = write_summary(results, arguments.output)
    print(f"Wrote {csv_path} and {markdown_path}")
    print(markdown_path.read_text(encoding="utf-8"))

    # Model bars use the final runs only, so each model is shown at its chosen learning rate.
    # Before any final run exists, fall back to all full-data runs (e.g. during the sweep).
    final_results = [result for result in results if is_final_run(result)]
    bar_results = final_results or [result for result in results if result.train_fraction == 1]
    charts = [
        plot_model_comparison(bar_results, "dev", arguments.output / "models_dev.png"),
        plot_learning_rate_sweep(results, arguments.output / "learning_rate_sweep.png"),
        plot_learning_curve(results, arguments.output / "learning_curve.png"),
    ]
    if arguments.include_test and final_results:
        charts.append(plot_model_comparison(final_results, "test", arguments.output / "models_test.png"))
    print("Charts:\n" + "\n".join(f"  {chart_path}" for chart_path in charts))


if __name__ == "__main__":
    main()

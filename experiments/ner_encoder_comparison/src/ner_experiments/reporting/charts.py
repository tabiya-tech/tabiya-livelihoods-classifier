"""Charts (matplotlib). Each function draws one chart and saves it as a PNG.

Which charts we keep is still open; add new ones here as plain functions taking typed
results and an output path.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only; no window needed
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from ner_experiments.schemas import RunResult, SplitScores  # noqa: E402
from ner_experiments.tags import EntityType  # noqa: E402


def _save(figure: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_training_history(result: RunResult, loss_curve: list[tuple[int, float]], output_path: Path) -> Path:
    """Training loss over update steps, and validation F1 after each pass."""
    figure, (loss_axes, score_axes) = plt.subplots(1, 2, figsize=(11, 4))
    if loss_curve:
        steps, losses = zip(*loss_curve)
        loss_axes.plot(steps, losses, linewidth=1.2)
    loss_axes.set_title("Training loss (lower is better)")
    loss_axes.set_xlabel("update step")
    loss_axes.set_ylabel("loss")

    passes = [record.epoch for record in result.history]
    dev_scores = [100 * (record.dev_macro_strict_f1 or 0) for record in result.history]
    score_axes.plot(passes, dev_scores, marker="o")
    score_axes.set_title("Validation macro exact-match F1")
    score_axes.set_xlabel("pass over the data (epoch)")
    score_axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    score_axes.set_ylabel("F1 (0 to 100)")
    figure.suptitle(result.run_name)
    return _save(figure, output_path)


def plot_scores_per_type(scores: SplitScores, title: str, output_path: Path) -> Path:
    """Precision, recall and F1 for each phrase type."""
    figure, axes = plt.subplots(figsize=(7, 4))
    entity_types = list(EntityType)
    bar_width = 0.25
    for offset_index, (metric_label, metric_name) in enumerate(
        [("precision", "precision"), ("recall", "recall"), ("F1", "f1")]
    ):
        values = [100 * getattr(scores.strict[entity_type], metric_name) for entity_type in entity_types]
        positions = [type_index + (offset_index - 1) * bar_width for type_index in range(len(entity_types))]
        axes.bar(positions, values, width=bar_width, label=metric_label)
    axes.set_xticks(range(len(entity_types)), [entity_type.value for entity_type in entity_types])
    axes.set_ylim(0, 100)
    axes.set_ylabel("score (0 to 100), exact match")
    axes.set_title(title)
    axes.legend()
    return _save(figure, output_path)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _mean_per_setting(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Average the scores of runs that share a setting (e.g. several seeds at one learning rate)."""
    scores_by_setting: dict[float, list[float]] = defaultdict(list)
    for setting, score in points:
        scores_by_setting[setting].append(score)
    return sorted((setting, _mean(scores)) for setting, scores in scores_by_setting.items())


def plot_model_comparison(results: list[RunResult], split: str, output_path: Path) -> Path:
    """Mean macro F1 per model across its runs, with the lowest and highest run as a range."""
    scores_by_model: dict[str, list[float]] = defaultdict(list)
    for result in results:
        split_scores = result.test if split == "test" else result.dev
        if split_scores is not None:
            scores_by_model[result.model_name].append(100 * split_scores.macro_strict_f1)
    model_names = sorted(scores_by_model, key=lambda name: -_mean(scores_by_model[name]))
    means = [_mean(scores_by_model[name]) for name in model_names]
    highest = [max(scores_by_model[name]) for name in model_names]
    lower_errors = [mean - min(scores_by_model[name]) for mean, name in zip(means, model_names)]
    upper_errors = [top - mean for top, mean in zip(highest, means)]

    figure, axes = plt.subplots(figsize=(8, 4.5))
    axes.bar(range(len(model_names)), means, yerr=[lower_errors, upper_errors], capsize=6)
    axes.set_xticks(range(len(model_names)), [name.split("/")[-1] for name in model_names], rotation=15)
    axes.set_ylabel(f"{split} macro exact-match F1")
    axes.set_ylim(0, max(highest, default=100) * 1.15)
    axes.set_title(f"Models on {split}: mean of runs, whiskers show lowest to highest run")
    for bar_index, (name, mean, top) in enumerate(zip(model_names, means, highest)):
        axes.annotate(
            f"{mean:.1f} (n={len(scores_by_model[name])})", (bar_index, top), xytext=(0, 4),
            textcoords="offset points", ha="center", va="bottom", fontsize=8,
        )
    return _save(figure, output_path)


def _plot_mean_lines(
    points_by_model: dict[str, list[tuple[float, float]]], axes: plt.Axes, minimum_settings: int
) -> None:
    drew_a_line = False
    for model_name, points in sorted(points_by_model.items()):
        averaged = _mean_per_setting(points)
        if len(averaged) < minimum_settings:
            continue
        (line,) = axes.plot(
            [setting for setting, _ in averaged], [score for _, score in averaged], marker="o",
            label=model_name.split("/")[-1],
        )
        axes.scatter(
            [setting for setting, _ in points], [score for _, score in points], alpha=0.35, s=14,
            color=line.get_color(),
        )
        drew_a_line = True
    if drew_a_line:
        axes.legend()
    else:
        axes.text(0.5, 0.5, "not enough runs yet", transform=axes.transAxes, ha="center")


def plot_learning_rate_sweep(results: list[RunResult], output_path: Path) -> Path:
    """Validation F1 against learning speed (learning rate): line = mean over seeds, dots = single runs."""
    points_by_model: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for result in results:
        if result.train_fraction == 1:
            points_by_model[result.model_name].append((result.learning_rate, 100 * result.dev.macro_strict_f1))
    figure, axes = plt.subplots(figsize=(7, 4))
    _plot_mean_lines(points_by_model, axes, minimum_settings=1)
    axes.set_xscale("log")
    learning_rates = sorted({rate for points in points_by_model.values() for rate, _ in points})
    axes.set_xticks(learning_rates, [f"{rate:.0e}" for rate in learning_rates])
    axes.minorticks_off()
    axes.set_xlabel("learning rate")
    axes.set_ylabel("dev macro exact-match F1")
    axes.set_title("Learning-rate sweep")
    return _save(figure, output_path)


def plot_learning_curve(results: list[RunResult], output_path: Path) -> Path:
    """Validation F1 against the share of training ads used: line = mean over seeds, dots = single runs."""
    points_by_model: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for result in results:
        points_by_model[result.model_name].append((100 * result.train_fraction, 100 * result.dev.macro_strict_f1))
    figure, axes = plt.subplots(figsize=(7, 4))
    _plot_mean_lines(points_by_model, axes, minimum_settings=2)
    axes.set_xlabel("share of training ads used (%)")
    axes.set_ylabel("dev macro exact-match F1")
    axes.set_title("Learning curve: does more labelled data still help?")
    return _save(figure, output_path)

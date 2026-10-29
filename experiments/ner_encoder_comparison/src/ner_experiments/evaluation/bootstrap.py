"""Is the gap between two models real, or luck? (paired bootstrap over ads)

Build many random versions of the test set by drawing ads with replacement, score every
model on each version (averaging its repeat runs), and look at the range of the gap
between models. When the middle 95% of that range includes zero, the models are tied.
Every model is scored on the same drawn ads, so hard ads affect all models equally.
"""

from collections import defaultdict
from itertools import combinations

import numpy as np
from pydantic import BaseModel

from ner_experiments.evaluation.counting import count_strict_matches
from ner_experiments.schemas import PredictedSentence
from ner_experiments.tags import ENTITY_TYPE_NAMES


class ModelSummary(BaseModel):
    model_name: str
    runs: int
    mean_macro_strict_f1: float
    interval_low: float
    interval_high: float


class PairwiseGap(BaseModel):
    better_model: str
    worse_model: str
    mean_gap: float
    interval_low: float
    interval_high: float
    tied: bool


class BootstrapResult(BaseModel):
    split: str
    resamples: int
    ads: int
    models: list[ModelSummary]
    gaps: list[PairwiseGap]


def _counts_per_ad(sentences: list[PredictedSentence], ad_ids: list[str]) -> np.ndarray:
    """Array shaped (ads, types, 3) holding correct / predicted / real counts."""
    sentences_by_ad: dict[str, list[PredictedSentence]] = defaultdict(list)
    for sentence in sentences:
        sentences_by_ad[sentence.ad_id].append(sentence)
    counts = np.zeros((len(ad_ids), len(ENTITY_TYPE_NAMES), 3), dtype=np.int64)
    for ad_position, ad_id in enumerate(ad_ids):
        ad_sentences = sentences_by_ad[ad_id]
        ad_counts = count_strict_matches(
            [sentence.tags for sentence in ad_sentences],
            [sentence.predicted_tags for sentence in ad_sentences],
        )
        for type_position, entity_type in enumerate(ENTITY_TYPE_NAMES):
            type_counts = ad_counts[entity_type]
            counts[ad_position, type_position] = (type_counts.correct, type_counts.predicted, type_counts.gold)
    return counts


def _macro_f1_from_totals(totals: np.ndarray) -> np.ndarray:
    """totals shaped (..., types, 3) -> macro F1 shaped (...)."""
    correct, predicted, gold = totals[..., 0], totals[..., 1], totals[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, correct / predicted, 0.0)
        recall = np.where(gold > 0, correct / gold, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return f1.mean(axis=-1)


def paired_bootstrap(
    predictions_by_model: dict[str, list[list[PredictedSentence]]],
    split: str,
    resamples: int = 1000,
    seed: int = 0,
) -> BootstrapResult:
    """``predictions_by_model`` maps a model name to the predictions of each of its runs."""
    first_run = next(iter(predictions_by_model.values()))[0]
    ad_ids = sorted({sentence.ad_id for sentence in first_run})
    random_generator = np.random.default_rng(seed)
    drawn_ads = random_generator.integers(0, len(ad_ids), size=(resamples, len(ad_ids)))

    resampled_f1_by_model: dict[str, np.ndarray] = {}
    for model_name, runs in predictions_by_model.items():
        per_run_f1 = []
        for run_predictions in runs:
            counts = _counts_per_ad(run_predictions, ad_ids)
            resampled_totals = counts[drawn_ads].sum(axis=1)
            per_run_f1.append(_macro_f1_from_totals(resampled_totals))
        resampled_f1_by_model[model_name] = np.mean(per_run_f1, axis=0)

    def summarise(model_name: str) -> ModelSummary:
        values = resampled_f1_by_model[model_name]
        return ModelSummary(
            model_name=model_name,
            runs=len(predictions_by_model[model_name]),
            mean_macro_strict_f1=float(values.mean()),
            interval_low=float(np.percentile(values, 2.5)),
            interval_high=float(np.percentile(values, 97.5)),
        )

    gaps: list[PairwiseGap] = []
    for first_model, second_model in combinations(resampled_f1_by_model, 2):
        gap_values = resampled_f1_by_model[first_model] - resampled_f1_by_model[second_model]
        if gap_values.mean() < 0:
            first_model, second_model, gap_values = second_model, first_model, -gap_values
        interval_low, interval_high = np.percentile(gap_values, [2.5, 97.5])
        gaps.append(
            PairwiseGap(
                better_model=first_model,
                worse_model=second_model,
                mean_gap=float(gap_values.mean()),
                interval_low=float(interval_low),
                interval_high=float(interval_high),
                tied=bool(interval_low <= 0 <= interval_high),
            )
        )

    models = sorted((summarise(name) for name in resampled_f1_by_model), key=lambda summary: -summary.mean_macro_strict_f1)
    return BootstrapResult(split=split, resamples=resamples, ads=len(ad_ids), models=models, gaps=gaps)

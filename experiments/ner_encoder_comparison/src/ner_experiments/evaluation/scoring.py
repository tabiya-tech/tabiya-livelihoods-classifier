"""Score predictions for one split: strict and partial F1 per type, and Skill phrase length.

The strict numbers are cross-checked against ``seqeval`` (the standard NER scorer, strict
mode), so they match what other papers report.
"""

import warnings

from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.scheme import IOB2

from ner_experiments.evaluation.counting import (
    CountsByType,
    count_partial_matches,
    count_strict_matches,
    macro_and_micro_f1,
    precision_recall_f1,
)
from ner_experiments.schemas import PredictedSentence, SpanLengthStats, SplitName, SplitScores, TypeScores
from ner_experiments.tags import EntityType, extract_spans

LONG_PHRASE_WORDS = 6


def _type_scores(counts_by_type: CountsByType) -> dict[EntityType, TypeScores]:
    scores: dict[EntityType, TypeScores] = {}
    for entity_type in EntityType:
        counts = counts_by_type[entity_type.value]
        precision, recall, f1 = precision_recall_f1(counts)
        scores[entity_type] = TypeScores(
            precision=precision, recall=recall, f1=f1,
            correct=counts.correct, predicted=counts.predicted, support=counts.gold,
        )
    return scores


def _skill_phrase_lengths(tag_sequences: list[list[str]]) -> list[int]:
    return [
        end_word - start_word
        for tags in tag_sequences
        for entity_type, start_word, end_word in extract_spans(tags)
        if entity_type == EntityType.SKILL.value
    ]


def _span_length_stats(gold_tag_sequences: list[list[str]], predicted_tag_sequences: list[list[str]]) -> SpanLengthStats:
    gold_lengths = _skill_phrase_lengths(gold_tag_sequences)
    predicted_lengths = _skill_phrase_lengths(predicted_tag_sequences)

    def mean(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    def share_long(values: list[int]) -> float:
        return sum(value >= LONG_PHRASE_WORDS for value in values) / len(values) if values else 0.0

    return SpanLengthStats(
        gold_mean_words=mean(gold_lengths),
        predicted_mean_words=mean(predicted_lengths),
        gold_share_six_plus_words=share_long(gold_lengths),
        predicted_share_six_plus_words=share_long(predicted_lengths),
    )


def score_split(predicted_sentences: list[PredictedSentence], split: SplitName) -> SplitScores:
    gold_tag_sequences = [sentence.tags for sentence in predicted_sentences]
    predicted_tag_sequences = [sentence.predicted_tags for sentence in predicted_sentences]

    strict_counts = count_strict_matches(gold_tag_sequences, predicted_tag_sequences)
    partial_counts = count_partial_matches(gold_tag_sequences, predicted_tag_sequences)
    macro_strict_f1, micro_strict_f1 = macro_and_micro_f1(strict_counts)
    macro_partial_f1, _micro_partial_f1 = macro_and_micro_f1(partial_counts)

    seqeval_macro_f1 = seqeval_f1_score(
        gold_tag_sequences, predicted_tag_sequences, mode="strict", scheme=IOB2, average="macro", zero_division=0
    )
    if abs(float(seqeval_macro_f1) - macro_strict_f1) > 1e-6:
        warnings.warn(
            f"strict macro F1 differs from seqeval: ours {macro_strict_f1:.6f}, seqeval {float(seqeval_macro_f1):.6f}",
            stacklevel=2,
        )

    return SplitScores(
        split=split,
        sentences=len(predicted_sentences),
        macro_strict_f1=macro_strict_f1,
        micro_strict_f1=micro_strict_f1,
        macro_partial_f1=macro_partial_f1,
        strict=_type_scores(strict_counts),
        partial=_type_scores(partial_counts),
        skill_span_length=_span_length_stats(gold_tag_sequences, predicted_tag_sequences),
    )


def print_split_scores(scores: SplitScores) -> None:
    print(f"\n{scores.split} set, {scores.sentences} sentences")
    print(f"{'type':<15}{'precision':>10}{'recall':>10}{'F1':>10}{'partial F1':>12}{'real':>8}")
    for entity_type in EntityType:
        strict = scores.strict[entity_type]
        partial = scores.partial[entity_type]
        print(
            f"{entity_type.value:<15}{strict.precision:>10.3f}{strict.recall:>10.3f}"
            f"{strict.f1:>10.3f}{partial.f1:>12.3f}{strict.support:>8}"
        )
    print(f"\nMacro exact-match F1 (the headline number): {100 * scores.macro_strict_f1:.1f}")
    print(f"Macro partial F1: {100 * scores.macro_partial_f1:.1f}")
    lengths = scores.skill_span_length
    print(
        f"Skill phrase length: real {lengths.gold_mean_words:.1f} words, predicted "
        f"{lengths.predicted_mean_words:.1f} words; {100 * lengths.predicted_share_six_plus_words:.1f}% "
        f"of predicted skills are {LONG_PHRASE_WORDS}+ words (real: {100 * lengths.gold_share_six_plus_words:.1f}%)"
    )

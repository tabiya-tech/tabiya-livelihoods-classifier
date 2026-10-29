"""Count correct, predicted and real phrases per type; turn counts into precision / recall / F1.

Two ways of deciding a predicted phrase is correct:

* strict (exact match): same type, same first word, same last word.
* partial (overlap): same type and at least one word in common. Each real phrase can be
  matched by at most one predicted phrase, so splitting one real phrase in two still
  counts one of the halves as a false alarm.

Counts are kept per type so they can be added up across sentences or ads, which is what
makes the bootstrap resampling fast.
"""

from dataclasses import dataclass

from ner_experiments.tags import ENTITY_TYPE_NAMES, extract_spans


@dataclass
class MatchCounts:
    correct: int = 0
    predicted: int = 0
    gold: int = 0

    def add(self, other: "MatchCounts") -> None:
        self.correct += other.correct
        self.predicted += other.predicted
        self.gold += other.gold


CountsByType = dict[str, MatchCounts]


def empty_counts() -> CountsByType:
    return {entity_type: MatchCounts() for entity_type in ENTITY_TYPE_NAMES}


def count_strict_matches(gold_tag_sequences: list[list[str]], predicted_tag_sequences: list[list[str]]) -> CountsByType:
    counts = empty_counts()
    for gold_tags, predicted_tags in zip(gold_tag_sequences, predicted_tag_sequences):
        gold_spans = set(extract_spans(gold_tags))
        predicted_spans = set(extract_spans(predicted_tags))
        for entity_type, *_ in gold_spans:
            counts[entity_type].gold += 1
        for span in predicted_spans:
            counts[span[0]].predicted += 1
            if span in gold_spans:
                counts[span[0]].correct += 1
    return counts


def count_partial_matches(gold_tag_sequences: list[list[str]], predicted_tag_sequences: list[list[str]]) -> CountsByType:
    counts = empty_counts()
    for gold_tags, predicted_tags in zip(gold_tag_sequences, predicted_tag_sequences):
        gold_spans = extract_spans(gold_tags)
        predicted_spans = extract_spans(predicted_tags)
        matched_gold_indices: set[int] = set()
        for entity_type, *_ in gold_spans:
            counts[entity_type].gold += 1
        for predicted_type, predicted_start, predicted_end in predicted_spans:
            counts[predicted_type].predicted += 1
            for gold_index, (gold_type, gold_start, gold_end) in enumerate(gold_spans):
                overlaps = predicted_start < gold_end and gold_start < predicted_end
                if gold_type == predicted_type and overlaps and gold_index not in matched_gold_indices:
                    matched_gold_indices.add(gold_index)
                    counts[predicted_type].correct += 1
                    break
    return counts


def precision_recall_f1(counts: MatchCounts) -> tuple[float, float, float]:
    precision = counts.correct / counts.predicted if counts.predicted else 0.0
    recall = counts.correct / counts.gold if counts.gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def macro_and_micro_f1(counts_by_type: CountsByType) -> tuple[float, float]:
    """Macro: average the three types' F1 equally. Micro: pool all phrases, then score."""
    per_type_f1 = [precision_recall_f1(counts)[2] for counts in counts_by_type.values()]
    macro_f1 = sum(per_type_f1) / len(per_type_f1)
    pooled = MatchCounts()
    for counts in counts_by_type.values():
        pooled.add(counts)
    return macro_f1, precision_recall_f1(pooled)[2]

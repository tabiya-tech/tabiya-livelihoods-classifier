"""Typed records passed between the stages: tagged sentences, predictions, scores, results."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ner_experiments.tags import TAG_NAMES, EntityType

SplitName = Literal["train", "dev", "test"]
_VALID_TAGS = frozenset(TAG_NAMES)


class TaggedSentence(BaseModel):
    """One sentence of a job ad, split into words, with one tag per word."""

    model_config = ConfigDict(frozen=True)

    ad_id: str = Field(description="Which ad the sentence comes from, e.g. 'random/262'")
    sentence_index: int = Field(ge=0, description="Position of the sentence in its ad")
    tokens: list[str] = Field(min_length=1, description="The sentence's words")
    tags: list[str] = Field(description="One tag per word: O, B-<type> or I-<type>")

    @model_validator(mode="after")
    def _one_valid_tag_per_word(self) -> "TaggedSentence":
        if len(self.tokens) != len(self.tags):
            raise ValueError(f"{len(self.tokens)} words but {len(self.tags)} tags")
        unknown_tags = set(self.tags) - _VALID_TAGS
        if unknown_tags:
            raise ValueError(f"unknown tags: {sorted(unknown_tags)}")
        return self


class PredictedSentence(TaggedSentence):
    """A tagged sentence plus the model's predicted tag for every word."""

    predicted_tags: list[str]

    @model_validator(mode="after")
    def _one_prediction_per_word(self) -> "PredictedSentence":
        if len(self.predicted_tags) != len(self.tokens):
            raise ValueError("predicted_tags must have one tag per word")
        return self


class TypeScores(BaseModel):
    """Precision, recall and F1 for one phrase type (or all types pooled)."""

    precision: float
    recall: float
    f1: float
    correct: int = Field(description="Predicted phrases that match a real one")
    predicted: int = Field(description="Phrases the model predicted")
    support: int = Field(description="Real phrases in the answer key")


class SpanLengthStats(BaseModel):
    """How long Skill phrases are, in words: the known over-long-span problem."""

    gold_mean_words: float
    predicted_mean_words: float
    gold_share_six_plus_words: float
    predicted_share_six_plus_words: float


class SplitScores(BaseModel):
    """All scores for one split (dev or test)."""

    split: SplitName
    sentences: int
    macro_strict_f1: float = Field(description="The headline number: exact-match F1 averaged over the three types")
    micro_strict_f1: float
    macro_partial_f1: float
    strict: dict[EntityType, TypeScores]
    partial: dict[EntityType, TypeScores]
    skill_span_length: SpanLengthStats


class EpochRecord(BaseModel):
    """What happened during one pass over the training data."""

    epoch: float
    step: int
    train_loss: float | None = None
    dev_macro_strict_f1: float | None = None


class RunResult(BaseModel):
    """Everything one experiment run produced, written to ``result.json``."""

    run_name: str
    model_name: str
    seed: int
    learning_rate: float
    train_fraction: float
    device: str
    precision: str
    train_minutes: float
    epochs_trained: float
    best_checkpoint_step: int | None
    training_sentences: int
    history: list[EpochRecord]
    dev: SplitScores
    test: SplitScores | None = Field(default=None, description="Only filled when evaluate_on_test is on")
    config: dict


class LatencyResult(BaseModel):
    """CPU speed of one trained model, one ad at a time."""

    model_dir: str
    threads: int
    ads_timed: int
    median_ms_per_ad: float
    p95_ms_per_ad: float
    peak_memory_mb: float
    parameters_millions: float

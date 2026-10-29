"""Fine-tune a pretrained model on the training sentences, choosing the best pass by the
validation (dev) score and stopping early when it stalls."""

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from ner_experiments.config import ExperimentConfig
from ner_experiments.evaluation.counting import macro_and_micro_f1, count_strict_matches
from ner_experiments.modeling.dataset import EncodedSentenceDataset
from ner_experiments.modeling.devices import DeviceName, PrecisionName
from ner_experiments.modeling.tokenization import IGNORED_IN_LOSS
from ner_experiments.progress import TrainingProgressCallback
from ner_experiments.schemas import EpochRecord
from ner_experiments.tags import TAG_ID_TO_NAME, TAG_NAME_TO_ID, TAG_NAMES, repair_bio


@dataclass
class TrainingOutcome:
    model: PreTrainedModel
    train_minutes: float
    epochs_trained: float
    best_checkpoint_step: int | None
    history: list[EpochRecord]


def load_model_for_tagging(model_name: str) -> PreTrainedModel:
    """The pretrained model with a fresh tagging layer for our seven tags on top."""
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(TAG_NAMES),
        id2label=TAG_ID_TO_NAME,
        label2id=TAG_NAME_TO_ID,
    )


def _dev_score_during_training(evaluation: EvalPrediction) -> dict[str, float]:
    """Validation score after each pass, computed on the labelled first pieces only."""
    best_tag_ids = np.argmax(evaluation.predictions, axis=-1)
    gold_tag_sequences: list[list[str]] = []
    predicted_tag_sequences: list[list[str]] = []
    for predicted_ids, gold_ids in zip(best_tag_ids, evaluation.label_ids):
        labelled_positions = gold_ids != IGNORED_IN_LOSS
        gold_tag_sequences.append([TAG_ID_TO_NAME[int(tag_id)] for tag_id in gold_ids[labelled_positions]])
        predicted_tag_sequences.append(
            repair_bio([TAG_ID_TO_NAME[int(tag_id)] for tag_id in predicted_ids[labelled_positions]])
        )
    match_counts = count_strict_matches(gold_tag_sequences, predicted_tag_sequences)
    macro_f1, micro_f1 = macro_and_micro_f1(match_counts)
    return {"macro_strict_f1": macro_f1, "micro_strict_f1": micro_f1}


def _history_from_log(log_history: list[dict]) -> list[EpochRecord]:
    """One record per evaluation (end of each pass), with the latest training loss."""
    records: list[EpochRecord] = []
    latest_train_loss: float | None = None
    for log_entry in log_history:
        if "loss" in log_entry:
            latest_train_loss = float(log_entry["loss"])
        if "eval_macro_strict_f1" in log_entry:
            records.append(
                EpochRecord(
                    epoch=float(log_entry.get("epoch", 0)),
                    step=int(log_entry.get("step", 0)),
                    train_loss=latest_train_loss,
                    dev_macro_strict_f1=float(log_entry["eval_macro_strict_f1"]),
                )
            )
    return records


def _training_loss_curve(log_history: list[dict]) -> list[tuple[int, float]]:
    return [(int(entry["step"]), float(entry["loss"])) for entry in log_history if "loss" in entry]


def train_model(
    settings: ExperimentConfig,
    tokenizer: PreTrainedTokenizerFast,
    train_dataset: EncodedSentenceDataset,
    dev_dataset: EncodedSentenceDataset,
    run_dir: Path,
    device: DeviceName,
    precision: PrecisionName,
) -> tuple[TrainingOutcome, list[tuple[int, float]]]:
    model = load_model_for_tagging(settings.model_name)
    checkpoints_dir = run_dir / "checkpoints"
    training_arguments = TrainingArguments(
        output_dir=str(checkpoints_dir),
        seed=settings.seed,
        learning_rate=settings.learning_rate,
        num_train_epochs=settings.max_epochs,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.batch_size * 2,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        warmup_ratio=settings.warmup_ratio,
        weight_decay=settings.weight_decay,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        use_cpu=device == "cpu",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_strict_f1",
        greater_is_better=True,
        logging_steps=25,
        report_to=[],
        dataloader_num_workers=settings.dataloader_workers,
        group_by_length=True,
        disable_tqdm=False,
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=_dev_score_during_training,
        callbacks=[
            TrainingProgressCallback(),
            EarlyStoppingCallback(early_stopping_patience=settings.early_stopping_patience),
        ],
    )

    training_start_time = time.time()
    trainer.train()
    train_minutes = (time.time() - training_start_time) / 60

    best_checkpoint = trainer.state.best_model_checkpoint
    best_checkpoint_step = int(best_checkpoint.rsplit("-", 1)[-1]) if best_checkpoint else None
    outcome = TrainingOutcome(
        model=trainer.model,
        train_minutes=train_minutes,
        epochs_trained=float(trainer.state.epoch or 0),
        best_checkpoint_step=best_checkpoint_step,
        history=_history_from_log(trainer.state.log_history),
    )
    loss_curve = _training_loss_curve(trainer.state.log_history)

    if settings.keep_model:
        trainer.save_model(str(run_dir / "model"))
        tokenizer.save_pretrained(str(run_dir / "model"))
    shutil.rmtree(checkpoints_dir, ignore_errors=True)
    return outcome, loss_curve

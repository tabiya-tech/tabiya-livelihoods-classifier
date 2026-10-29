"""Train and score ONE run, as set by a config file (and any environment overrides).

    python -m ner_experiments.cli.run_experiment --config configs/runs/xlmr-base.yaml
    NER_EXP_SEED=2 python -m ner_experiments.cli.run_experiment --config configs/runs/xlmr-base.yaml

The config file can also come from the NER_EXP_CONFIG_FILE environment variable.
Results land in runs/<run name>/ (see run_files.py for what each file holds).
"""

import argparse
import os
import shutil
from pathlib import Path

from transformers import set_seed

from ner_experiments.config import ExperimentConfig, load_settings
from ner_experiments.corpus.loading import keep_share_of_ads, read_sentences
from ner_experiments.evaluation.scoring import print_split_scores, score_split
from ner_experiments.modeling.dataset import EncodedSentenceDataset
from ner_experiments.modeling.devices import describe_device, resolve_device, resolve_precision
from ner_experiments.modeling.prediction import predict_sentences
from ner_experiments.modeling.tokenization import load_tokenizer
from ner_experiments.modeling.training import train_model
from ner_experiments.progress import StageReporter
from ner_experiments.reporting.charts import plot_scores_per_type, plot_training_history
from ner_experiments.run_files import (
    RESULT_FILE,
    write_effective_config,
    write_loss_curve,
    write_predictions,
    write_result,
)
from ner_experiments.schemas import RunResult


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=os.environ.get("NER_EXP_CONFIG_FILE"))
    arguments = parser.parse_args()

    stages = StageReporter(total_stages=7)

    stages.start("Settings")
    settings = load_settings(ExperimentConfig, Path(arguments.config) if arguments.config else None)
    run_dir = settings.run_dir()
    if (run_dir / RESULT_FILE).exists() and not settings.overwrite:
        raise SystemExit(f"{run_dir} already has a finished run; set overwrite: true or pick another run_name")
    if run_dir.exists() and settings.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_effective_config(run_dir, settings.model_dump(mode="json"))
    set_seed(settings.seed)
    device = resolve_device(settings.device)
    precision = resolve_precision(settings.precision, device)
    print(f"Run: {settings.resolved_run_name()}  ->  {run_dir}")
    print(f"Model: {settings.model_name}")
    print(f"Device: {describe_device(device)}, number format: {precision}")
    print(
        f"Learning rate {settings.learning_rate:g}, seed {settings.seed}, batch {settings.batch_size}, "
        f"up to {settings.max_epochs} passes, {100 * settings.train_fraction:.0f}% of training ads, "
        f"test set {'ON' if settings.evaluate_on_test else 'off (locked)'}"
    )

    stages.start("Loading the train / dev / test sentences")
    train_sentences = keep_share_of_ads(
        read_sentences(settings.paths.processed_dir, "train"), settings.train_fraction, settings.seed
    )
    dev_sentences = read_sentences(settings.paths.processed_dir, "dev")
    test_sentences = read_sentences(settings.paths.processed_dir, "test") if settings.evaluate_on_test else []
    print(f"{len(train_sentences)} training, {len(dev_sentences)} dev, {len(test_sentences)} test sentences")

    stages.start("Loading the tokenizer and splitting words into pieces")
    tokenizer = load_tokenizer(settings.model_name)
    train_dataset = EncodedSentenceDataset(
        train_sentences, tokenizer, settings.max_pieces_per_sentence, "Encoding training sentences"
    )
    dev_dataset = EncodedSentenceDataset(dev_sentences, tokenizer, settings.max_pieces_per_sentence, "Encoding dev sentences")
    print(
        f"{len(train_dataset)} training chunks ({train_dataset.chunked_sentence_count} long sentences split), "
        f"{len(dev_dataset)} dev chunks"
    )

    stages.start("Training (stops early when the validation score stalls)")
    outcome, loss_curve = train_model(settings, tokenizer, train_dataset, dev_dataset, run_dir, device, precision)
    print(f"Trained {outcome.epochs_trained:.0f} passes in {outcome.train_minutes:.1f} min; best step {outcome.best_checkpoint_step}")

    stages.start("Predicting dev" + (" and test" if settings.evaluate_on_test else ""))
    dev_predictions = predict_sentences(
        outcome.model, tokenizer, dev_sentences, settings.max_pieces_per_sentence, device, description="Predicting dev"
    )
    write_predictions(run_dir, "dev", dev_predictions)
    test_predictions = []
    if settings.evaluate_on_test:
        test_predictions = predict_sentences(
            outcome.model, tokenizer, test_sentences, settings.max_pieces_per_sentence, device, description="Predicting test"
        )
        write_predictions(run_dir, "test", test_predictions)

    stages.start("Scoring")
    dev_scores = score_split(dev_predictions, "dev")
    print_split_scores(dev_scores)
    test_scores = score_split(test_predictions, "test") if settings.evaluate_on_test else None
    if test_scores is not None:
        print_split_scores(test_scores)

    stages.start("Saving results and charts")
    result = RunResult(
        run_name=settings.resolved_run_name(),
        model_name=settings.model_name,
        seed=settings.seed,
        learning_rate=settings.learning_rate,
        train_fraction=settings.train_fraction,
        device=device,
        precision=precision,
        train_minutes=outcome.train_minutes,
        epochs_trained=outcome.epochs_trained,
        best_checkpoint_step=outcome.best_checkpoint_step,
        training_sentences=len(train_sentences),
        history=outcome.history,
        dev=dev_scores,
        test=test_scores,
        config=settings.model_dump(mode="json"),
    )
    write_result(run_dir, result)
    write_loss_curve(run_dir, loss_curve)
    charts_dir = run_dir / "charts"
    plot_training_history(result, loss_curve, charts_dir / "training_history.png")
    plot_scores_per_type(dev_scores, f"{result.run_name}: dev", charts_dir / "dev_scores_per_type.png")
    if test_scores is not None:
        plot_scores_per_type(test_scores, f"{result.run_name}: test", charts_dir / "test_scores_per_type.png")
    print(f"Saved result.json, predictions and charts to {run_dir}")
    stages.finish()


if __name__ == "__main__":
    main()

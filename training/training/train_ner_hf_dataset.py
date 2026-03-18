"""NER fine-tuning from a HuggingFace dataset (migrated from train/train.py).

This script trains against a HuggingFace-hosted dataset (e.g. tabiya/job_ner_dataset)
and optionally uses a CRF decoder.

Usage::

    python -m training.train_ner_hf_dataset \\
        --dataset tabiya/job_ner_dataset \\
        --output ./model-output \\
        --base-model bert-base-cased \\
        --epochs 4 \\
        --batch-size 32
"""

import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from nervaluate import Evaluator as NERvaluateEvaluator
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

import evaluate as hf_evaluate

LABEL_LIST = [
    "O",
    "B-Skill", "I-Skill",
    "B-Occupation", "I-Occupation",
    "B-Qualification", "I-Qualification",
    "B-Experience", "I-Experience",
    "B-Domain", "I-Domain",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tabiya/job_ner_dataset")
    parser.add_argument("--output", default="./results")
    parser.add_argument("--base-model", default="bert-base-cased")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--crf", action="store_true", help="Use CRF decoder")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, token=args.hf_token)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, model_max_length=128, add_prefix_space=True, token=args.hf_token
    )

    use_crf = args.crf
    pad_label_id = len(LABEL_LIST) if use_crf else -100

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        all_labels = []
        all_special_masks = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev = None
            label_ids = []
            special_mask = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(len(LABEL_LIST) if use_crf else -100)
                    if use_crf:
                        special_mask.append(0)
                elif word_idx != prev:
                    label_ids.append(label[word_idx])
                    if use_crf:
                        special_mask.append(0)
                else:
                    label_ids.append(len(LABEL_LIST) if use_crf else -100)
                    if use_crf:
                        special_mask.append(1)
                prev = word_idx
            all_labels.append(label_ids)
            if use_crf:
                all_special_masks.append(special_mask)
        tokenized["labels"] = all_labels
        if use_crf:
            tokenized["special_tokens_mask"] = all_special_masks
        return tokenized

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

    if use_crf:
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding="max_length",
            max_length=128,
            label_pad_token_id=len(LABEL_LIST),
        )
    else:
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    seqeval = hf_evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        ignore_id = len(LABEL_LIST) if use_crf else -100
        true_predictions = [
            [LABEL_LIST[pred] for pred, lbl in zip(prediction, label) if lbl != ignore_id]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [LABEL_LIST[lbl] for pred, lbl in zip(prediction, label) if lbl != ignore_id]
            for prediction, label in zip(predictions, labels)
        ]
        evaluator = NERvaluateEvaluator(
            true_labels,
            true_predictions,
            tags=["Skill", "Qualification", "Domain", "Experience", "Occupation"],
            loader="list",
        )
        evaluator.evaluate()
        results = seqeval.compute(
            predictions=true_predictions, references=true_labels, mode="strict", scheme="IOB2"
        )
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    if use_crf:
        from shared.transformers_crf import BertCrfForNer
        model = BertCrfForNer.from_pretrained(args.base_model, num_labels=len(LABEL_LIST) + 1)
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            args.base_model,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
            token=args.hf_token,
        )

    training_args = TrainingArguments(
        output_dir=args.output,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        optim="adamw_torch",
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate(tokenized_dataset["test"])
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()

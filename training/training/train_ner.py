"""NER fine-tuning script.

Data format (JSONL, one record per line)::

    {"tokens": ["Head", "Chef", "needed"], "ner_tags": ["B-Occupation", "I-Occupation", "O"]}

Usage::

    python -m training.train_ner \\
        --data train.jsonl \\
        --output ./model-output \\
        --base-model tabiya/roberta-base-job-ner \\
        --epochs 3 \\
        --batch-size 16
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


# BIO label set for job NER
LABEL_LIST = [
    "O",
    "B-Occupation", "I-Occupation",
    "B-Skill", "I-Skill",
    "B-Qualification", "I-Qualification",
    "B-Experience", "I-Experience",
    "B-Domain", "I-Domain",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def tokenize_and_align_labels(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(LABEL2ID.get(labels[word_idx], 0))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Fine-tune NER model")
    parser.add_argument("--data", required=True, help="Path to training JSONL file")
    parser.add_argument("--output", required=True, help="Output directory for model")
    parser.add_argument(
        "--base-model",
        default="tabiya/roberta-base-job-ner",
        help="Base model to fine-tune",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    args = parser.parse_args()

    print(f"Loading data from {args.data}")
    records = load_jsonl(args.data)
    dataset = Dataset.from_list(records)

    print(f"Loading tokenizer and model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=args.hf_token)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        token=args.hf_token,
        ignore_mismatched_sizes=True,
    )

    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()

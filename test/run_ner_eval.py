"""
Run NER evaluation on green_test.json using the same model as inference (default: tabiya/roberta-base-job-ner).
Writes span F1 strict metrics to test/eval_baseline_roberta.json for comparison with an LLM later.

Usage:
  poetry run python test/run_ner_eval.py           # full test set (can be slow on CPU)
  poetry run python test/run_ner_eval.py --limit 50   # first 50 samples only
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForTokenClassification, AutoTokenizer

load_dotenv()

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from seqeval.scheme import IOB2

import evaluate

ENTITY_MODEL = os.getenv("ENTITY_MODEL", "tabiya/roberta-base-job-ner")
TEST_FILE = Path(__file__).resolve().parent / "green_test.json"
OUTPUT_FILE = Path(__file__).resolve().parent / "eval_baseline_roberta.json"


def fix_bio_tags(tags: list) -> list:
    fixed = list(tags)
    for i in range(len(tags) - 2):
        if tags[i].startswith("B-") and tags[i + 1] == "O" and tags[i + 2].startswith("I-"):
            fixed[i + 1] = tags[i + 2]
        if tags[i] == "O" and tags[i + 1].startswith("I-") and tags[i + 2] == "O":
            fixed[i + 1] = "O"
    if len(tags) >= 2 and tags[-2] == "O" and tags[-1].startswith("I-"):
        fixed[-1] = "O"
    return fixed


def main():
    parser = argparse.ArgumentParser(description="Run NER evaluation on green_test.json")
    parser.add_argument("--limit", type=int, default=None, help="Max number of samples (default: all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        ENTITY_MODEL, token=os.getenv("HF_TOKEN"), add_prefix_space=True
    )
    model = AutoModelForTokenClassification.from_pretrained(
        ENTITY_MODEL, token=os.getenv("HF_TOKEN")
    )
    model.to(device)
    model.eval()

    id2label = model.config.id2label
    seqeval = evaluate.load("seqeval")

    lines = TEST_FILE.read_text().strip().split("\n")
    if args.limit:
        lines = lines[: args.limit]
    all_predictions = []
    all_references = []

    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        tokens = row["tokens"]
        gold = row["tags_skill"]

        enc = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            return_tensors="pt",
            padding=False,
        )
        word_ids = enc.word_ids(0)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits[0]
        pred_ids = logits.argmax(dim=1).cpu().tolist()
        pred_tags = [id2label.get(i, "O") for i in pred_ids]
        pred_tags = fix_bio_tags(pred_tags)

        input_ids = enc["input_ids"][0].cpu().tolist()
        special_ids = set(tokenizer.all_special_ids)
        word_id_and_tag = [
            (word_ids[i], pred_tags[i])
            for i in range(len(input_ids))
            if input_ids[i] not in special_ids
        ]
        if not word_id_and_tag:
            all_predictions.append([])
            all_references.append(gold[:0])
            continue

        pred_per_word = []
        seen = set()
        for w, t in word_id_and_tag:
            if w is not None and w not in seen:
                seen.add(w)
                pred_per_word.append((w, t))
        pred_per_word.sort(key=lambda x: x[0])
        pred_labels = [t for _, t in pred_per_word]

        n_words = len(pred_labels)
        ref_labels = gold[:n_words]

        all_predictions.append(pred_labels)
        all_references.append(ref_labels)

    results = seqeval.compute(
        predictions=all_predictions,
        references=all_references,
        mode="strict",
        scheme=IOB2,
    )

    out = {
        "model": ENTITY_MODEL,
        "test_file": str(TEST_FILE),
        "num_samples": len(all_predictions),
        "limit": args.limit,
        "metrics": {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        },
        "per_entity": {
            k: v for k, v in results.items()
            if k not in ("overall_precision", "overall_recall", "overall_f1", "overall_accuracy")
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"Results written to {OUTPUT_FILE}")
    print(f"F1: {out['metrics']['f1']:.4f}  P: {out['metrics']['precision']:.4f}  R: {out['metrics']['recall']:.4f}")


if __name__ == "__main__":
    main()

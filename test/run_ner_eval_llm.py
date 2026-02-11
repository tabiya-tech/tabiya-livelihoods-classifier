"""
Evaluate LLM-based NER (Vertex AI / Gemini) on green_test.json using the SAME
BIO-tag / span-F1 metric as the transformer baseline.

- Ground truth: test/green_test.json (tokens + tags_skill BIO labels)
- Backend: inference.ner_llm.VertexNERClient (Vertex AI Gemini)
- Output: test/eval_llm_<model>.json

Usage examples (from repo root):

  # Use env LLM_MODEL (default gemini-1.5-pro)
  poetry run python test/run_ner_eval_llm.py

  # Explicit Gemini 2.5 model
  poetry run python test/run_ner_eval_llm.py --llm-model gemini-2.5-flash

Required env for Vertex:
  - GOOGLE_APPLICATION_CREDENTIALS
  - VERTEX_PROJECT
  - (optional) VERTEX_API_REGION (default us-west1)
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

import evaluate

load_dotenv()

project_root = Path(__file__).resolve().parent.parent
TEST_FILE = Path(__file__).resolve().parent / "green_test.json"
OUTPUT_DIR = Path(__file__).resolve().parent


def build_suffix_map(labels: List[List[str]]) -> dict:
    """
    Infer dataset tag suffixes (e.g. Skill vs SKILL) from green_test.json.
    Returns mapping from canonical type names to the suffix actually used.
    """
    suffixes = set()
    for seq in labels:
        for tag in seq:
            if tag != "O" and "-" in tag:
                _, suf = tag.split("-", 1)
                suffixes.add(suf)
    suffix_map = {}
    for suf in suffixes:
        low = suf.lower()
        for canon in ["Skill", "Occupation", "Qualification", "Domain", "Experience"]:
            if low == canon.lower():
                suffix_map[canon] = suf
    return suffix_map


def align_spans_to_bio(tokens: List[str], entities: List[dict], suffix_map: dict) -> List[str]:
    """
    Convert LLM span output into token-level BIO tags, to match tags_skill.
    - tokens: original token list from green_test.json
    - entities: [{"type": "Skill", "tokens": "plan menus"}, ...]
    """
    n = len(tokens)
    pred = ["O"] * n

    lower_tokens = [t.lower() for t in tokens]

    for ent in entities:
        ent_type = ent.get("type")
        span_text = (ent.get("tokens") or "").strip()
        if not ent_type or not span_text:
            continue

        # Map canonical type to dataset suffix (e.g. Skill -> SKILL)
        suffix = suffix_map.get(ent_type)
        if not suffix:
            continue

        span_tokens = span_text.split()
        span_lower = [s.lower() for s in span_tokens]
        L = len(span_lower)
        if L == 0 or L > n:
            continue

        # Find first window that matches span text (case-insensitive)
        found = False
        for start in range(0, n - L + 1):
            window = lower_tokens[start : start + L]
            if window == span_lower:
                b_tag = f"B-{ent_type}"
                i_tag = f"I-{ent_type}"
                # Only overwrite if currently O to avoid conflicts
                if pred[start] == "O":
                    b_tag = f"B-{suffix}"
                    i_tag = f"I-{suffix}"
                    pred[start] = b_tag
                    for j in range(1, L):
                        if start + j < n and pred[start + j] == "O":
                            pred[start + j] = i_tag
                found = True
                break

        # If no match, we skip this span
        if not found:
            continue

    return pred


def main():
    parser = argparse.ArgumentParser(description="Run LLM NER evaluation on green_test.json")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of samples (default: all)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Vertex model id (overrides LLM_MODEL env, e.g. gemini-2.5-flash)",
    )
    args = parser.parse_args()

    # Import here so RoBERTa-only users don't need Vertex deps
    from inference.ner_llm import VertexNERClient

    llm_model = args.llm_model or os.getenv("LLM_MODEL", "gemini-1.5-pro")
    client = VertexNERClient(model_name=llm_model)

    seqeval = evaluate.load("seqeval")

    # Support both JSONL (one object per line) and a JSON array of objects
    content = TEST_FILE.read_text().strip()
    rows = []
    if content.startswith("["):
        rows = json.loads(content)
    else:
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if args.limit:
        rows = rows[: args.limit]

    # Preload gold labels to build suffix map
    gold_all = [row["tags_skill"] for row in rows]
    suffix_map = build_suffix_map(gold_all)

    all_predictions: List[List[str]] = []
    all_references: List[List[str]] = []
    detailed = []

    print(f"Using LLM model: {llm_model}")

    for row in tqdm(rows, desc="Evaluating LLM NER", unit="sent"):
        tokens = row["tokens"]
        gold = row["tags_skill"]

        sentence = " ".join(tokens)
        entities = client.extract(sentence)
        pred_labels = align_spans_to_bio(tokens, entities, suffix_map)

        # Ensure lengths match; if not, truncate gold to prediction length
        n = min(len(pred_labels), len(gold))
        all_predictions.append(pred_labels[:n])
        all_references.append(gold[:n])

        detailed.append(
            {
                "idx": row.get("idx"),
                "tokens": tokens,
                "gold_tags": gold,
                "llm_entities": entities,
                "pred_tags": pred_labels,
            }
        )

    results = seqeval.compute(
        predictions=all_predictions,
        references=all_references,
        mode="strict",
        scheme="IOB2",
    )

    safe_name = llm_model.replace("/", "_").replace(":", "_")
    out_name = f"eval_llm_{safe_name}.json"
    output_file = OUTPUT_DIR / out_name
    dump_name = f"llm_ner_outputs_{safe_name}.jsonl"
    dump_file = OUTPUT_DIR / dump_name

    out = {
        "backend": "llm",
        "llm_model": llm_model,
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
            k: v
            for k, v in results.items()
            if k
            not in (
                "overall_precision",
                "overall_recall",
                "overall_f1",
                "overall_accuracy",
            )
        },
    }

    # Convert numpy scalars to plain Python types for JSON
    def _to_python(o):
        if isinstance(o, (np.generic,)):
            return o.item()
        if isinstance(o, dict):
            return {k: _to_python(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_to_python(v) for v in o]
        return o

    output_file.write_text(json.dumps(_to_python(out), indent=2))
    # Write per-sentence detailed outputs for visual inspection
    with dump_file.open("w", encoding="utf-8") as f:
        for rec in detailed:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Results written to {output_file}")
    print(f"Detailed LLM NER outputs written to {dump_file}")
    print(
        f"LLM={llm_model}  F1: {out['metrics']['f1']:.4f}  "
        f"P: {out['metrics']['precision']:.4f}  R: {out['metrics']['recall']:.4f}"
    )


if __name__ == "__main__":
    main()


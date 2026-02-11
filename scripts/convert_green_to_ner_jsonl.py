"""
Convert green_test-style JSON (array of {tokens, tags_skill}) to the
sentence+entities JSONL format expected by prepare_gemma_ner_dataset.py.

Use this when you have jobs_ner_green.json from run_ner_on_jobs.py.

Usage:
  python scripts/convert_green_to_ner_jsonl.py \
    --input data/jobs_ner_green.json \
    --output data/pseudo_labeled_ner.jsonl
"""

import argparse
import json
from pathlib import Path

WANTED_TYPES = {"Occupation", "Skill", "Qualification"}


def bio_to_entities(tokens: list, tags: list) -> list:
    """Convert BIO tags to list of {type, tokens}."""
    entities = []
    i = 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            etype = tags[i][2:]
            if etype not in WANTED_TYPES:
                i += 1
                continue
            span = [tokens[i]]
            i += 1
            while i < len(tags) and tags[i] == f"I-{etype}":
                span.append(tokens[i])
                i += 1
            entities.append({"type": etype, "tokens": " ".join(span)})
        else:
            i += 1
    return entities


def main():
    parser = argparse.ArgumentParser(description="Convert green JSON to sentence+entities JSONL")
    parser.add_argument("--input", type=str, default="data/jobs_ner_green.json", help="Green-format JSON (array of {tokens, tags_skill})")
    parser.add_argument("--output", type=str, default="data/pseudo_labeled_ner.jsonl", help="Output JSONL for prepare_gemma_ner_dataset")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap number of examples")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        rows = [rows]
    if args.max_samples:
        rows = rows[: args.max_samples]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            tokens = row.get("tokens") or []
            tags = row.get("tags_skill") or []
            sentence = " ".join(tokens)
            entities = bio_to_entities(tokens, tags)
            f.write(json.dumps({"sentence": sentence, "entities": entities}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} examples to {out_path}")


if __name__ == "__main__":
    main()

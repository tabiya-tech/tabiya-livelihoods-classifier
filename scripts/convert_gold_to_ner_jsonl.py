"""
Convert the gold NER dataset (tabiya/job_ner_dataset) to the same sentence+entities
JSONL format as pseudo_label_job_descriptions.py output. Use this for the
optional second fine-tuning stage (fine-tune on gold after pseudo-labels).

Output can be fed to prepare_gemma_ner_dataset.py, then finetune_gemma_ner.py.

Usage:
  python scripts/convert_gold_to_ner_jsonl.py --output data/gold_ner.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

WANTED_TYPES = {"Occupation", "Skill", "Qualification"}


def main():
    parser = argparse.ArgumentParser(description="Convert gold NER HF dataset to sentence+entities JSONL")
    parser.add_argument("--output", type=str, default="data/gold_ner.jsonl", help="Output JSONL path")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/test)")
    parser.add_argument("--dataset", type=str, default="tabiya/job_ner_dataset", help="HuggingFace dataset path")
    args = parser.parse_args()

    from datasets import load_dataset
    from inference.linker import EntityLinker

    token = os.getenv("HF_TOKEN")
    ds = load_dataset(args.dataset, token=token, split=args.split)
    label_list = ds.features["ner_tags"].feature.names
    id2label = {i: name for i, name in enumerate(label_list)}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            tokens = ex["tokens"]
            tag_ids = ex["ner_tags"]
            tags = [id2label[i] for i in tag_ids]
            entities = EntityLinker.extract_entities(tokens, tags)
            entities_filtered = [
                {"type": e["type"], "tokens": " ".join(e["tokens"])}
                for e in entities
                if e["type"] in WANTED_TYPES
            ]
            sentence = " ".join(tokens)
            f.write(json.dumps({"sentence": sentence, "entities": entities_filtered}, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} examples to {out_path}")


if __name__ == "__main__":
    main()

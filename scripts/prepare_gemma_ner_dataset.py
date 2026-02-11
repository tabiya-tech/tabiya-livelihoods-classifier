"""
Convert pseudo-labeled or gold NER JSONL (sentence + entities) into SFT format
(prompt + response) for Gemma NER fine-tuning. Uses the same prompt template
as inference/gemma_ner.py so the model learns the exact task format.

Input JSONL: {"sentence": "...", "entities": [{"type": "...", "tokens": "..."}, ...]}
Output JSONL: {"prompt": "...", "response": "{\"entities\": [...]}"}
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from inference.gemma_ner import PROMPT_TEMPLATE


def main():
    parser = argparse.ArgumentParser(description="Prepare Gemma NER SFT dataset from sentence+entities JSONL")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL (sentence, entities)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL (prompt, response)")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap number of examples")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if args.max_samples and n >= args.max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sentence = obj.get("sentence", "").strip()
            entities = obj.get("entities") or []
            response_entities = [
                {"text": e.get("tokens", ""), "type": e.get("type", "")}
                for e in entities
                if e.get("type") in ("Occupation", "Skill", "Qualification")
            ]
            response_str = json.dumps({"entities": response_entities}, ensure_ascii=False)
            prompt = PROMPT_TEMPLATE.format(sentence=sentence)
            fout.write(json.dumps({"prompt": prompt, "response": response_str}, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} examples to {out_path}")


if __name__ == "__main__":
    main()

"""
Label job descriptions with the existing BERT/RoBERTa NER model (no linking).
Outputs JSONL: one object per sentence with "sentence" and "entities" for use in
prepare_gemma_ner_dataset.py and Gemma NER fine-tuning.

Usage:
  python scripts/pseudo_label_job_descriptions.py \
    --input path/to/job_descriptions.csv \
    --output data/pseudo_labeled_ner.jsonl \
    --max-descriptions 1000 \
    --text-column description
"""

import argparse
import json
import sys
from pathlib import Path

from nltk.tokenize import sent_tokenize

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_descriptions(path: Path, text_column: str, max_descriptions: int):
    path = Path(path)
    texts = []
    if path.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not in {list(df.columns)}")
        texts = df[text_column].dropna().astype(str).tolist()
    elif path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                t = obj.get("text") or obj.get("description") or obj.get("body")
                if t:
                    texts.append(str(t))
    else:
        raw = path.read_text(encoding="utf-8")
        for block in raw.split("\n\n"):
            block = block.replace("\n", " ").strip()
            if block:
                texts.append(block)
    if max_descriptions:
        texts = texts[:max_descriptions]
    return texts


def main():
    parser = argparse.ArgumentParser(description="Pseudo-label job descriptions with BERT NER")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV with job descriptions")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path (sentence + entities)")
    parser.add_argument("--max-descriptions", type=int, default=1000, help="Max number of job descriptions to process")
    parser.add_argument("--text-column", type=str, default="description", help="CSV column containing job description text")
    parser.add_argument("--entity-model", type=str, default="tabiya/roberta-base-job-ner", help="HuggingFace NER model (BERT/RoBERTa)")
    args = parser.parse_args()

    from inference.linker import EntityLinker

    linker = EntityLinker(
        entity_model=args.entity_model,
        ner_type="roberta",
        from_cache=True,
    )
    descriptions = load_descriptions(Path(args.input), args.text_column, args.max_descriptions)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for desc in descriptions:
            for sent in sent_tokenize(desc.replace("\n", " ").strip()):
                if not sent.strip():
                    continue
                entities = linker(sent, linking=False)
                entities_serial = [{"type": e["type"], "tokens": e["tokens"]} for e in entities]
                f.write(json.dumps({"sentence": sent, "entities": entities_serial}, ensure_ascii=False) + "\n")
                count += 1
    print(f"Wrote {count} sentence-level examples to {out_path}")


if __name__ == "__main__":
    main()

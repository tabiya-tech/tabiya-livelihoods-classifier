"""
Run BERT/RoBERTa NER on job descriptions from jobs-plain.json.
Reads description_plain or description_html, strips HTML, normalizes text,
breaks into paragraphs then sentences, and writes green_test_en.json-style output.

Usage:
  python scripts/run_ner_on_jobs.py --input jobs-plain.json --output data/jobs_ner_green.json
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path

import nltk
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm

nltk.download("punkt", quiet=True)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

WANTED_TYPES = {"Occupation", "Skill", "Qualification"}


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities. Block tags become newlines for paragraph breaks."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|tr|h[1-6])>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single space, strip, normalize newlines."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def get_clean_description(job: dict) -> str:
    """Prefer description_plain; if empty or looks like HTML, use description_html or raw.description and strip."""
    raw = job.get("description_plain") or job.get("description_html") or ""
    r = job.get("raw") or {}
    if not raw and isinstance(r, dict):
        raw = r.get("description") or ""
    if not raw:
        return ""
    if "<" in raw and ">" in raw:
        raw = strip_html(raw)
    return normalize_whitespace(raw)


def split_paragraphs(text: str, max_chars: int | None = None) -> list[str]:
    """Split on double newline. If max_chars set, long paragraphs are split into chunks by sentences."""
    if not text.strip():
        return []
    parts = re.split(r"\n\s*\n", text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if max_chars and len(p) > max_chars:
            sentences = sent_tokenize(p)
            chunk = ""
            for s in sentences:
                if chunk and len(chunk) + 1 + len(s) > max_chars:
                    out.append(chunk.strip())
                    chunk = s
                else:
                    chunk = f"{chunk} {s}".strip() if chunk else s
            if chunk:
                out.append(chunk)
        else:
            out.append(p)
    return out


def tokens_to_bio(tokens: list, entities: list) -> list:
    """Map entity spans (type + token string) to BIO tags aligned to tokens."""
    n = len(tokens)
    tags = ["O"] * n
    if not tokens:
        return tags
    lower_tokens = [t.lower() for t in tokens]

    for ent in entities:
        etype = ent.get("type")
        span = (ent.get("tokens") or "").strip()
        if etype not in WANTED_TYPES or not span:
            continue
        entity_words = re.sub(r"\s+", " ", span).strip().split()
        if not entity_words:
            continue
        entity_lower = [w.lower() for w in entity_words]
        L = len(entity_lower)
        if L > n:
            continue
        start_i = None
        for i in range(0, n - L + 1):
            if lower_tokens[i : i + L] == entity_lower:
                start_i = i
                break
        if start_i is None:
            continue
        end_i = start_i + L - 1
        suffix = etype
        for k in range(start_i, end_i + 1):
            if tags[k] == "O":
                tags[k] = f"B-{suffix}" if k == start_i else f"I-{suffix}"
    return tags


def _ner_extract_only(sentence: str, model, tokenizer, device) -> list:
    """Run NER only (no linker embeddings). Returns list of {type, tokens}."""
    import torch
    from inference.linker import EntityLinker

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
    predicted_token_class = EntityLinker.fix_bio_tags(predicted_token_class)
    input_ids, predicted_token_class = EntityLinker.remove_special_tokens_and_tags(
        inputs["input_ids"][0].tolist(), predicted_token_class, tokenizer
    )
    result = EntityLinker.extract_entities(
        input_ids, predicted_token_class
    )
    for entry in result:
        entry["tokens"] = tokenizer.decode(entry["tokens"]).lstrip(" ")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run NER on jobs-plain.json, output green_test-style JSON")
    parser.add_argument("--input", type=str, default="jobs-plain.json", help="Path to jobs-plain.json (array of jobs)")
    parser.add_argument("--output", type=str, default="data/jobs_ner_green.json", help="Output JSON path (green_test_en format)")
    parser.add_argument("--max-jobs", type=int, default=None, help="Max job descriptions to process (default: all)")
    parser.add_argument("--max-paragraph-chars", type=int, default=2000, help="Split paragraphs longer than this into chunks (0 = no limit)")
    parser.add_argument("--entity-model", type=str, default="tabiya/roberta-base-job-ner", help="HuggingFace NER model")
    args = parser.parse_args()

    import os
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    token = os.getenv("HF_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.entity_model, token=token)
    model = AutoModelForTokenClassification.from_pretrained(args.entity_model, token=token)
    model.to(device)

    with open(args.input, encoding="utf-8") as f:
        jobs = json.load(f)
    if not isinstance(jobs, list):
        jobs = [jobs]
    if args.max_jobs:
        jobs = jobs[: args.max_jobs]

    max_para = args.max_paragraph_chars or None
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = []
    idx = 1
    pbar = tqdm(desc="NER", unit="sent", dynamic_ncols=True)
    for job in jobs:
        text = get_clean_description(job)
        if not text:
            continue
        for block in split_paragraphs(text, max_chars=max_para):
            for sent in sent_tokenize(block):
                if not sent.strip():
                    continue
                tokens = word_tokenize(sent)
                if not tokens:
                    continue
                entities = _ner_extract_only(sent, model, tokenizer, device)
                tags_skill = tokens_to_bio(tokens, entities)
                result.append({"idx": idx, "tokens": tokens, "tags_skill": tags_skill})
                idx += 1
                pbar.update(1)
    pbar.close()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(result)} sentences to {out_path}")


if __name__ == "__main__":
    main()

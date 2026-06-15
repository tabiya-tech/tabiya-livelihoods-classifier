#!/usr/bin/env python3
"""Bake NEL model at build time. NER is copied in via ner_model_cache/ (run prepare_ner_for_bake.sh first)."""
import os

# NEL model (SentenceTransformer)
print("Downloading NEL model (all-MiniLM-L6-v2)...")
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
print("NEL model cached")

# NER: already in /app/.cache/huggingface from COPY ner_model_cache, or try HF download
ner_cache = "/app/.cache/huggingface/hub/models--tabiya--roberta-base-job-ner"
if os.path.isdir(ner_cache):
    print("NER model already in image (from ner_model_cache)")
else:
    hf_token = (os.environ.get("HF_TOKEN") or "").strip()
    if hf_token:
        print("Downloading NER model (tabiya/roberta-base-job-ner)...")
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer
            AutoModelForTokenClassification.from_pretrained(
                "tabiya/roberta-base-job-ner", token=hf_token
            )
            AutoTokenizer.from_pretrained(
                "tabiya/roberta-base-job-ner", token=hf_token
            )
            print("NER model cached")
        except Exception as e:
            print(f"WARNING: NER model bake failed ({e}) — run prepare_ner_for_bake.sh first")
    else:
        print("WARNING: NER not in image. Run: ./scripts/prepare_ner_for_bake.sh before build")

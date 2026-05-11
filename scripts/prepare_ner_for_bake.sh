#!/bin/sh
# Copy locally-downloaded NER model into build context for baking into Docker image.
# Run once before: docker build -f docker/Dockerfile.job -t tabiya-classifier:latest .
set -e

CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
SRC="$CACHE/hub/models--tabiya--roberta-base-job-ner"
DST="ner_model_cache/hub/models--tabiya--roberta-base-job-ner"

if [ ! -d "$SRC" ]; then
  echo "Model not found at $SRC"
  echo "Run first: hf download tabiya/roberta-base-job-ner"
  exit 1
fi

mkdir -p ner_model_cache/hub
rm -rf "$DST"
cp -r "$SRC" "$DST"
echo "Copied NER model to $DST"
echo "Now run: docker build -f docker/Dockerfile.job -t tabiya-classifier:latest ."

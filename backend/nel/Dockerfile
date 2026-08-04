# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.3

# Copy shared package first (local path dependency)
COPY backend/shared /app/shared

# Copy NEL service
COPY backend/nel /app/nel

WORKDIR /app/nel

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without dev

# Bake the models into the image at build time so startup is fast and never
# hits HuggingFace Hub (avoids 429s on shared Cloud Run IPs). One per language:
# English uses all-MiniLM-L6-v2, other languages a multilingual model — see
# shared/languages/<lang>_config.py.
RUN --mount=type=secret,id=HF_TOKEN \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN) \
    huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
    && huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Languages this image serves. `all` = every entry in shared.languages.LANGUAGE_REGISTRY;
# their linkers are loaded at startup rather than on the first request that needs them.
ENV ENABLED_LANGUAGES="all"
# Default language for requests that don't carry one.
ENV TARGET_LANGUAGE="en"
ENV MAX_ENTITIES_PER_REQUEST="200"
ENV MAX_TOP_K="50"
ENV HF_HUB_OFFLINE="1"
# NEL_FILES_PATH is deliberately NOT set: it pins every language to a single directory,
# and the data packs live per language under /app/nel/nel/files/<lang>/.

# Pre-compute each language pack's embeddings at build time so cold starts are instant.
RUN python - <<'EOF'
from shared.languages import LANGUAGE_REGISTRY, get_language_config
from nel.linker import NELLinker, resolve_files_path

for language in LANGUAGE_REGISTRY:
    model = get_language_config(language)["nel_similarity_model"]
    path = resolve_files_path(language)
    print(f"Pre-computing {language} embeddings ({model}) from {path}…", flush=True)
    NELLinker(similarity_model=model, files_path=path, from_cache=False, language=language)
print("Embeddings pre-computed and cached.")
EOF

EXPOSE 5003

CMD ["uvicorn", "nel.main:app", "--host", "0.0.0.0", "--port", "5003"]

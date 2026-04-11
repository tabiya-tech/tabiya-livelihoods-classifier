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

# Bake the model into the image at build time so startup is fast and never
# hits HuggingFace Hub (avoids 429s on shared Cloud Run IPs).
RUN --mount=type=secret,id=HF_TOKEN \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN) \
    huggingface-cli download sentence-transformers/all-MiniLM-L6-v2

ENV LINKER_MODEL="all-MiniLM-L6-v2"
ENV MAX_ENTITIES_PER_REQUEST="200"
ENV MAX_TOP_K="50"
ENV HF_HUB_OFFLINE="1"
ENV NEL_FILES_PATH="/app/nel/nel/files"

# Pre-compute embeddings at build time so cold starts are instant.
RUN python - <<'EOF'
from nel.linker import NELLinker
NELLinker(similarity_model="all-MiniLM-L6-v2", files_path="/app/nel/nel/files", from_cache=False)
print("Embeddings pre-computed and cached.")
EOF

EXPOSE 5003

CMD ["uvicorn", "nel.main:app", "--host", "0.0.0.0", "--port", "5003"]

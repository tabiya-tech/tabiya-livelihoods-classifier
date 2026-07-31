# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.3

# Copy shared package first (local path dependency)
COPY backend/shared /app/shared

# Copy NEL v2 service
COPY backend/nel_v2 /app/nel_v2

WORKDIR /app/nel_v2

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without dev --extras inference

# Bake all supported SentenceTransformer models into the image so cold starts
# don't hit HuggingFace Hub. Vertex AI models (text-embedding-005,
# gemini-embedding-001) are remote-only and don't need caching here.
RUN --mount=type=secret,id=HF_TOKEN \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN) \
    huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 && \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN) \
    huggingface-cli download sentence-transformers/paraphrase-multilingual-mpnet-base-v2 && \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN) \
    huggingface-cli download tabiya/all-MiniLM-L6-v2-occupation-fine-tuned

ENV HF_HUB_OFFLINE="1"

EXPOSE 5003

CMD ["uvicorn", "nel.main:app", "--host", "0.0.0.0", "--port", "5003"]

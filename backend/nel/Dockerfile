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

# Copy taxonomy reference files (CSV + embeddings)
# These should be volume-mounted or baked in during build
ENV LINKER_MODEL="all-MiniLM-L6-v2"
ENV MAX_ENTITIES_PER_REQUEST="200"
ENV MAX_TOP_K="50"

EXPOSE 5003

CMD ["uvicorn", "nel.main:app", "--host", "0.0.0.0", "--port", "5003"]

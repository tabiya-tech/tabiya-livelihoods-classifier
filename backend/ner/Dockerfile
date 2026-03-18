FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Copy shared package first (local path dependency)
COPY backend/shared /app/shared

# Copy NER service
COPY backend/ner /app/ner

WORKDIR /app/ner

# Install dependencies (shared is installed as a local path dep)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without dev

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

ENV NER_MODEL="tabiya/roberta-base-job-ner"
ENV MAX_TEXT_LENGTH="50000"

EXPOSE 5002

CMD ["uvicorn", "ner.main:app", "--host", "0.0.0.0", "--port", "5002"]

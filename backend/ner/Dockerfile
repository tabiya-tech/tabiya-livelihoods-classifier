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

# Download NLTK data. punkt_tab carries the per-language sentence tokenizers the
# service selects from (see shared/languages/<lang>_config.py:sentence_tokenizer_language).
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Languages this image serves. `all` = every entry in shared.languages.LANGUAGE_REGISTRY,
# loaded at startup rather than on the first request that needs them.
ENV ENABLED_LANGUAGES="all"
# Default language for requests that don't carry one.
ENV TARGET_LANGUAGE="en"
# NER_MODEL is deliberately NOT set: the bare variable applies to every language, which
# would force a non-English language onto the English checkpoint. Per-language defaults
# live in shared/languages/<lang>_config.py; override one with NER_MODEL_<LANG>.
ENV MAX_TEXT_LENGTH="50000"

EXPOSE 5002

CMD ["uvicorn", "ner.main:app", "--host", "0.0.0.0", "--port", "5002"]

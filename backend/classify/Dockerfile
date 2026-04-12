FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.3

# Copy shared package first (local path dependency)
COPY backend/shared /app/shared

# Copy classify service
COPY backend/classify /app/classify

WORKDIR /app/classify

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without dev

ENV NER_API_URL="http://ner-service"
ENV NEL_API_URL="http://nel-service"
ENV CLASSIFIER_VERSION="1.0.0"
ENV MAX_TEXT_LENGTH="50000"
ENV MAX_BATCH_SIZE="500"

EXPOSE 5001

CMD ["uvicorn", "classify.main:app", "--host", "0.0.0.0", "--port", "5001"]

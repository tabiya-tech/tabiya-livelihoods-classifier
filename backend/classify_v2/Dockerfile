FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.3

# Copy classify v2 service
COPY backend/classify_v2 /app/classify_v2

WORKDIR /app/classify_v2

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without dev

EXPOSE 5004

CMD ["uvicorn", "classify_v2.main:app", "--host", "0.0.0.0", "--port", "5004"]

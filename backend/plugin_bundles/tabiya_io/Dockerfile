FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.3

COPY backend/tabiya_plugin_contracts /app/tabiya_plugin_contracts

COPY backend/plugin_bundles/tabiya_io /app/tabiya_io

WORKDIR /app/tabiya_io

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without dev

EXPOSE 5011

CMD ["uvicorn", "tabiya_io.main:app", "--host", "0.0.0.0", "--port", "5011"]

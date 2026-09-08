FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.3

# Copy shared contracts package first (local path dependency).
# Must land at /tabiya_plugin_contracts so that the relative path
# "../../tabiya_plugin_contracts" from /app/tabiya_core resolves correctly.
COPY backend/tabiya_plugin_contracts /tabiya_plugin_contracts

# Copy the bundle
COPY backend/plugin_bundles/tabiya_core /app/tabiya_core

WORKDIR /app/tabiya_core

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without dev

EXPOSE 5010

# Invoke via `python -m uvicorn` so startup does not depend on the `uvicorn`
# console script being on PATH (it isn't reliably under poetry's system install).
CMD ["python", "-m", "uvicorn", "tabiya_core.main:app", "--host", "0.0.0.0", "--port", "5010"]

FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry lock && \
    poetry install --no-root --no-interaction --no-ansi && \
    python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

COPY . .

EXPOSE 5001 5002 5003

CMD ["python", "app/server/classify_server.py"]

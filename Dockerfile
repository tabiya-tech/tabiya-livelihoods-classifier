FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

COPY . .

EXPOSE 5001 5002 5003

CMD ["uvicorn", "app.server.classify_server:app", "--host", "0.0.0.0", "--port", "5001"]

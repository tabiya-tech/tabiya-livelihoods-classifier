#!/bin/sh
# Cloud Run *Service*: NER + NEL in background, Classify API on :5001.
# Same image as docker/Dockerfile.job; Job uses entrypoint.sh (runs run_classifier.py).
set -e

NER_PID=""
NEL_PID=""
CLASSIFY_PID=""

cleanup() {
  kill "$CLASSIFY_PID" "$NER_PID" "$NEL_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

forward_term() {
  kill -TERM "$CLASSIFY_PID" "$NER_PID" "$NEL_PID" 2>/dev/null || true
}
trap forward_term TERM

echo "Starting NER API (background)..."
python app/server/ner_server.py &
NER_PID=$!

echo "Starting NEL API (background)..."
python app/server/nel_server.py &
NEL_PID=$!

echo "Waiting for NER and NEL APIs..."
for i in $(seq 1 90); do
  if python -c "
import urllib.request
try:
  urllib.request.urlopen('http://localhost:5002/v1/health', timeout=2)
  urllib.request.urlopen('http://localhost:5003/v1/health', timeout=2)
except: raise SystemExit(1)
" 2>/dev/null; then
    echo "NER and NEL ready"
    break
  fi
  sleep 2
  if [ "$i" -eq 90 ]; then
    echo "Timeout waiting for NER/NEL APIs"
    exit 1
  fi
done

export NER_API_URL=http://localhost:5002
export NEL_API_URL=http://localhost:5003

echo "Starting Classify API on 0.0.0.0:5001..."
python app/server/classify_server.py &
CLASSIFY_PID=$!

echo "Waiting for Classify API..."
for i in $(seq 1 30); do
  if python -c "
import urllib.request
try:
  urllib.request.urlopen('http://localhost:5001/v1/health', timeout=2)
except: raise SystemExit(1)
" 2>/dev/null; then
    echo "Classify API ready"
    break
  fi
  sleep 2
  if [ "$i" -eq 30 ]; then
    echo "Timeout waiting for Classify API"
    exit 1
  fi
done

wait "$CLASSIFY_PID"

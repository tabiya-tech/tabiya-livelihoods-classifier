#!/bin/sh
# All-in-one classifier: start NER, NEL, Classify APIs, then run run_classifier.py
set -e

echo "Starting NER API (background)..."
python app/server/ner_server.py &
NER_PID=$!

echo "Starting NEL API (background)..."
python app/server/nel_server.py &
NEL_PID=$!

# Wait for NER and NEL to be ready (model load can take 2+ min on cold start)
echo "Waiting for NER and NEL APIs..."
for i in $(seq 1 90); do
  if python -c "
import urllib.request
try:
  urllib.request.urlopen('http://localhost:5002/v1/health', timeout=2)
  urllib.request.urlopen('http://localhost:5003/v1/health', timeout=2)
except: exit(1)
" 2>/dev/null; then
    echo "NER and NEL ready"
    break
  fi
  sleep 2
  if [ $i -eq 90 ]; then
    echo "Timeout waiting for NER/NEL APIs"
    kill $NER_PID $NEL_PID 2>/dev/null || true
    exit 1
  fi
done

echo "Starting Classify API (background)..."
export NER_API_URL=http://localhost:5002
export NEL_API_URL=http://localhost:5003
python app/server/classify_server.py &
CLASSIFY_PID=$!

# Wait for Classify API
echo "Waiting for Classify API..."
for i in $(seq 1 30); do
  if python -c "
import urllib.request
try:
  urllib.request.urlopen('http://localhost:5001/v1/health', timeout=2)
except: exit(1)
" 2>/dev/null; then
    echo "Classify API ready"
    break
  fi
  sleep 2
  if [ $i -eq 30 ]; then
    echo "Timeout waiting for Classify API"
    kill $NER_PID $NEL_PID $CLASSIFY_PID 2>/dev/null || true
    exit 1
  fi
done

# Run the classifier job (reads MongoDB, calls classify API, writes results)
export CLASSIFY_API_URL=http://localhost:5001
echo "Running classifier job..."
python run_classifier.py
EXIT_CODE=$?

# Cleanup
kill $CLASSIFY_PID $NER_PID $NEL_PID 2>/dev/null || true
exit $EXIT_CODE

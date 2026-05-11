#!/usr/bin/env bash
# Start NER (5002) + NEL (5003) + Classify (5001) locally for testing.
# Requires: cd .. && pip install deps from pyproject; HF_TOKEN in .env for NER model.
#
# Fixes macOS NLTK punkt issues by using repo-local NLTK_DATA:
#   ./scripts/start_local_classifier_stack.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
NLTKD="${ROOT}/.nltk_data"
export NLTK_DATA="${NLTK_DATA:-$NLTKD}"
if [[ ! -d "$NLTK_DATA/tokenizers/punkt_tab" ]]; then
  echo "Downloading NLTK punkt into $NLTK_DATA ..."
  export SSL_CERT_FILE="$(python3 -c "import certifi; print(certifi.where())")"
  python3 -c "import nltk, os; d=os.environ['NLTK_DATA']; nltk.download('punkt_tab', download_dir=d, quiet=True); nltk.download('punkt', download_dir=d, quiet=True)"
fi
[[ -f .env ]] && set -a && source .env && set +a
export NER_API_URL="${NER_API_URL:-http://127.0.0.1:5002}"
export NEL_API_URL="${NEL_API_URL:-http://127.0.0.1:5003}"

echo "Starting ner_server :5002 ..."
nohup env NLTK_DATA="$NLTK_DATA" python3 app/server/ner_server.py >> /tmp/tabiya-ner.log 2>&1 &
echo "Starting nel_server :5003 ..."
nohup env NLTK_DATA="$NLTK_DATA" python3 app/server/nel_server.py >> /tmp/tabiya-nel.log 2>&1 &
sleep 3
echo "Starting classify_server :5001 ..."
nohup env NLTK_DATA="$NLTK_DATA" NER_API_URL="$NER_API_URL" NEL_API_URL="$NEL_API_URL" \
  python3 app/server/classify_server.py >> /tmp/tabiya-classify.log 2>&1 &
sleep 2
curl -s http://127.0.0.1:5001/v1/health | python3 -m json.tool || true
echo "CLASSIFY_API_URL=http://127.0.0.1:5001"
echo "Logs: /tmp/tabiya-ner.log /tmp/tabiya-nel.log /tmp/tabiya-classify.log"

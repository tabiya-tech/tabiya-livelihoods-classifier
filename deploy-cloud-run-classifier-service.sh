#!/usr/bin/env bash
# Deploy the Tabiya classifier as a Cloud Run *Service* (HTTP Classify API on port 5001).
# One container runs NER :5002 + NEL :5003 + Classify orchestrator :5001 (see docker/entrypoint-service.sh).
# The batch Job uses the same image with ./docker/entrypoint.sh (runs run_classifier.py).
#
# Build image first (needs HF_TOKEN in Secret Manager for Cloud Build):
#   cd "/path/to/Demand Side"
#   gcloud builds submit --config=tabiya-livelihoods-classifier/cloudbuild.yaml .
#
# Deploy:
#   chmod +x tabiya-livelihoods-classifier/deploy-cloud-run-classifier-service.sh
#   ./tabiya-livelihoods-classifier/deploy-cloud-run-classifier-service.sh
#
# Smoke test (public — default deploy):
#   URL=$(gcloud run services describe livelihoods-classifier-api --region=us-central1 --format='value(status.url)')
#   curl -sS "${URL}/v1/health"
#   curl -sS -X POST -H "Content-Type: application/json" \
#     -d '{"text":"Head Chef with menu planning."}' "${URL}/v1/classify"
#
# If you deployed with CLASSIFIER_REQUIRE_IAM=1, add:
#   -H "Authorization: Bearer $(gcloud auth print-identity-token)"
#
set -euo pipefail

PROJECT="${GCP_PROJECT:-horizon-dev-481316}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${CLASSIFIER_SERVICE_NAME:-livelihoods-classifier-api}"
IMAGE="${CLASSIFIER_IMAGE:-us-central1-docker.pkg.dev/${PROJECT}/job-pipeline/classifier:service}"

# Flask Classify app listens here (must match classify_server.py).
CONTAINER_PORT=5001

# Optional secrets (usually not required — models are baked in Dockerfile.job).
SECRETS="${CLASSIFIER_SERVICE_SECRETS:-}"

# Public ingress (no Google ID token): anyone with the URL can call the API.
# The app has no separate client API key — use CLASSIFIER_REQUIRE_IAM=1 if you need IAM-only.
if [[ "${CLASSIFIER_REQUIRE_IAM:-0}" == "1" ]]; then
  AUTH_FLAG=(--no-allow-unauthenticated)
else
  AUTH_FLAG=(--allow-unauthenticated)
fi

gcloud run deploy "${SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --image="${IMAGE}" \
  --platform=managed \
  --port="${CONTAINER_PORT}" \
  --cpu=4 \
  --memory=8Gi \
  --timeout=3600 \
  --cpu-boost \
  --concurrency="${CLASSIFIER_CONCURRENCY:-2}" \
  --min-instances="${CLASSIFIER_MIN_INSTANCES:-0}" \
  --max-instances="${CLASSIFIER_MAX_INSTANCES:-10}" \
  --no-cpu-throttling \
  --command=./docker/entrypoint-service.sh \
  ${SECRETS:+--set-secrets="${SECRETS}"} \
  "${AUTH_FLAG[@]}"

echo ""
echo "Deployed: ${SERVICE}"
gcloud run services describe "${SERVICE}" --project="${PROJECT}" --region="${REGION}" \
  --format='value(status.url)'

#!/usr/bin/env bash
# upload_config.sh — upload all local config files for an environment to GCP Secret Manager.
#
# Usage:
#   ./iac/scripts/upload_config.sh dev tabiya-classifier-dev
#
# Prerequisites:
#   - gcloud auth login (or GOOGLE_APPLICATION_CREDENTIALS set)
#   - gcloud config set project <project>

set -euo pipefail

STACK="${1:?Usage: $0 <stack> <gcp-project>}"
PROJECT="${2:?Usage: $0 <stack> <gcp-project>}"
CONFIG_DIR="$(cd "$(dirname "$0")/../config/${STACK}" && pwd)"

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "error: config directory not found: $CONFIG_DIR" >&2
  exit 1
fi

upload_secret() {
  local secret_id="$1"
  local file="$2"

  if [[ ! -f "$file" ]]; then
    echo "  skipping '$secret_id' — file not found: $file"
    return
  fi

  # Create secret if it doesn't exist
  if ! gcloud secrets describe "$secret_id" --project "$PROJECT" &>/dev/null; then
    echo "  creating secret '$secret_id'..."
    gcloud secrets create "$secret_id" --project "$PROJECT" --replication-policy automatic
  fi

  echo "  uploading '$secret_id' from $(basename "$file")..."
  gcloud secrets versions add "$secret_id" --project "$PROJECT" --data-file="$file"
}

echo "Uploading config for stack '$STACK' to project '$PROJECT'..."
echo

upload_secret "env-vars"                      "$CONFIG_DIR/env-vars"
upload_secret "stack-config-enable-services"  "$CONFIG_DIR/stack-config-enable-services.yaml"
upload_secret "stack-config-dns"              "$CONFIG_DIR/stack-config-dns.yaml"
upload_secret "stack-config-auth"             "$CONFIG_DIR/stack-config-auth.yaml"
upload_secret "stack-config-backend"          "$CONFIG_DIR/stack-config-backend.yaml"
upload_secret "stack-config-common"           "$CONFIG_DIR/stack-config-common.yaml"
upload_secret "stack-config-aws-ns"           "$CONFIG_DIR/stack-config-aws-ns.yaml"

echo
echo "Done. Verify with:"
echo "  gcloud secrets list --project $PROJECT"

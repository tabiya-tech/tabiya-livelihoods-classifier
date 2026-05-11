"""Runtime configuration for the classify service."""

import os
from dotenv import load_dotenv

load_dotenv()

NER_API_URL = os.getenv("NER_API_URL", "http://localhost:5002")
NEL_API_URL = os.getenv("NEL_API_URL", "http://localhost:5003")
CLASSIFIER_VERSION = os.getenv("CLASSIFIER_VERSION", "1.0.0")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "50000"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "500"))

# Max concurrent classify pipelines (single /v1/classify + batch jobs combined).
# Caps load on NER/NEL and avoids unbounded parallelism when many batches run at once.
MAX_CONCURRENT_CLASSIFY_OPS = max(1, int(os.getenv("MAX_CONCURRENT_CLASSIFY_OPS", "20")))

MONGODB_URI = os.getenv("APPLICATION_MONGODB_URI", "")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "tabiya-classifier")

# When true, Motor accepts invalid TLS certificates (legacy behaviour). Default false
# for production; set MONGODB_TLS_ALLOW_INVALID=true only if Atlas/CI requires it.
def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


MONGODB_TLS_ALLOW_INVALID = _env_bool("MONGODB_TLS_ALLOW_INVALID", False)

API_KEY_CACHE_TTL = int(os.getenv("API_KEY_CACHE_TTL", "300"))  # 5 min

# Firebase project ID — used in the API Gateway spec for token verification.
# The gateway verifies the token and forwards decoded user info to the backend.
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")

# GCP project and managed service name — used to create GCP-managed API keys
# that are validated by the API Gateway.
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GCP_API_MANAGED_SERVICE = os.getenv("GCP_API_MANAGED_SERVICE", "")

# Set to "local" in local development so Bearer tokens are decoded without
# signature verification (mirrors the Compass pattern).
TARGET_ENVIRONMENT_TYPE = os.getenv("TARGET_ENVIRONMENT_TYPE", "")

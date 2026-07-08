import os
from dotenv import load_dotenv

load_dotenv()

NER_API_URL: str = os.getenv("NER_API_URL", "http://localhost:5002")
NEL_V2_API_URL: str = os.getenv("NEL_V2_API_URL", "http://localhost:5003")

# Plugin bundle URLs — resolved by the pipeline registry to fetch manifests
# and (post-11.6) invoke plugin stages. Unset bundles are surfaced to the
# frontend as UNAVAILABLE per design §10.
TABIYA_CORE_BUNDLE_URL: str = os.getenv("TABIYA_CORE_BUNDLE_URL", "")
TABIYA_IO_BUNDLE_URL: str = os.getenv("TABIYA_IO_BUNDLE_URL", "")

# Defaults used when seeding the Default Tabiya pipeline for a fresh user.
# Both must be set for lazy seeding to activate; leaving them empty means
# a fresh user's `GET /v2/pipelines` returns an empty list until an admin
# configures them (or the user creates their own pipeline).
DEFAULT_NEL_MODEL_ID: str = os.getenv("DEFAULT_NEL_MODEL_ID", "")
DEFAULT_TAXONOMY_MODEL_ID: str = os.getenv("DEFAULT_TAXONOMY_MODEL_ID", "")
CLASSIFIER_VERSION: str = os.getenv("CLASSIFIER_VERSION", "2.0.0")
MAX_TEXT_LENGTH: int = int(os.getenv("MAX_TEXT_LENGTH", "50000"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
TARGET_ENVIRONMENT_TYPE: str = os.getenv("TARGET_ENVIRONMENT_TYPE", "")
CORS_ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

# Application MongoDB — stores api_keys collection (and future user_configs etc.)
APPLICATION_MONGODB_URI: str = os.getenv("APPLICATION_MONGODB_URI", "")
APPLICATION_DATABASE_NAME: str = os.getenv("APPLICATION_DATABASE_NAME", "tabiya-classifier")

# Firebase + GCP API Keys provisioning. ADC via GOOGLE_APPLICATION_CREDENTIALS.
FIREBASE_PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "")
GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "")
GCP_API_MANAGED_SERVICE: str = os.getenv("GCP_API_MANAGED_SERVICE", "")
GCP_API_KEYS_PARENT_LOCATION: str = os.getenv("GCP_API_KEYS_PARENT_LOCATION", "global")

# Hard upper bound on keys per user (frontend disables Create when reached).
MAX_API_KEYS_PER_USER: int = int(os.getenv("MAX_API_KEYS_PER_USER", "5"))

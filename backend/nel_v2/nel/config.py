import os

from dotenv import load_dotenv

load_dotenv()

# Application DB — user configs, nel_models, nel_qualifications, cache status
APPLICATION_MONGODB_URI: str = os.getenv("APPLICATION_MONGODB_URI", "")
APPLICATION_DATABASE_NAME: str = os.getenv("APPLICATION_DATABASE_NAME", "tabiya-classifier")

# Taxonomy DB — Atlas cluster with vector search indexes for embeddings
TAXONOMY_MONGODB_URI: str = os.getenv("TAXONOMY_MONGODB_URI", "")
TAXONOMY_DATABASE_NAME: str = os.getenv("TAXONOMY_DATABASE_NAME", "tabiya-taxonomy")

# Taxonomy REST API
TAXONOMY_API_BASE_URL: str = os.getenv("TAXONOMY_API_BASE_URL", "https://taxonomy.tabiya.tech")
TAXONOMY_API_KEY: str = os.getenv("TAXONOMY_API_KEY", "")

# Default model configuration (overridden by user config in classify-v2)
DEFAULT_NEL_MODEL_ID: str = os.getenv("DEFAULT_NEL_MODEL_ID", "all-MiniLM-L6-v2")
DEFAULT_TAXONOMY_MODEL_ID: str = os.getenv("DEFAULT_TAXONOMY_MODEL_ID", "")

# Service metadata
CLASSIFIER_VERSION: str = os.getenv("CLASSIFIER_VERSION", "2.0.0")

# Logging
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# Environment
TARGET_ENVIRONMENT_TYPE: str = os.getenv("TARGET_ENVIRONMENT_TYPE", "")

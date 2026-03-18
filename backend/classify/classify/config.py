"""Runtime configuration for the classify service."""

import os
from dotenv import load_dotenv

load_dotenv()

NER_API_URL = os.getenv("NER_API_URL", "http://localhost:5002")
NEL_API_URL = os.getenv("NEL_API_URL", "http://localhost:5003")
CLASSIFIER_VERSION = os.getenv("CLASSIFIER_VERSION", "1.0.0")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "50000"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "500"))

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_BATCH_TTL = int(os.getenv("REDIS_BATCH_TTL", "3600"))  # 1 hour

MONGODB_URI = os.getenv("APPLICATION_MONGODB_URI", "")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "tabiya-classifier")

API_KEY_CACHE_TTL = int(os.getenv("API_KEY_CACHE_TTL", "300"))  # 5 min

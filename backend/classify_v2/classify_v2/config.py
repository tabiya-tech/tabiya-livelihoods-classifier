import os
from dotenv import load_dotenv

load_dotenv()

NER_API_URL: str = os.getenv("NER_API_URL", "http://localhost:5002")
NEL_V2_API_URL: str = os.getenv("NEL_V2_API_URL", "http://localhost:5003")
CLASSIFIER_VERSION: str = os.getenv("CLASSIFIER_VERSION", "2.0.0")
MAX_TEXT_LENGTH: int = int(os.getenv("MAX_TEXT_LENGTH", "50000"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
TARGET_ENVIRONMENT_TYPE: str = os.getenv("TARGET_ENVIRONMENT_TYPE", "")
CORS_ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

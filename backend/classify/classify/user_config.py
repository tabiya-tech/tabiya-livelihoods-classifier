"""User configuration resolution: API key → user_id → config.

API Gateway forwards the raw X-Api-Key header; classify resolves it to a user
config via MongoDB, with an in-process cache backed by Redis.
"""

import hashlib
import logging
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection

from classify.config import (
    API_KEY_CACHE_TTL,
    MONGODB_DB_NAME,
    MONGODB_URI,
)

log = logging.getLogger("classify-api")

_mongo_client: Optional[MongoClient] = None
_in_memory_cache: dict = {}  # key_hash -> (user_config, expires_at)


def _get_collection(name: str) -> Collection:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGODB_URI)
    db = _mongo_client[MONGODB_DB_NAME]
    return db[name]


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def get_user_config(api_key: str) -> Optional[dict]:
    """Resolve an API key to user config.

    Returns a dict with keys: user_id, ner_model_name, nel_model_name,
    taxonomy_model_id, ner_type, nel_type.
    Returns None if the key is unknown.

    Falls back to default config if MongoDB is not configured.
    """
    if not MONGODB_URI:
        return _default_config()

    import time

    key_hash = _hash_key(api_key)

    cached = _in_memory_cache.get(key_hash)
    if cached:
        config, expires_at = cached
        if time.time() < expires_at:
            return config
        del _in_memory_cache[key_hash]

    try:
        api_keys_col = _get_collection("api_keys")
        key_doc = api_keys_col.find_one({"key_hash": key_hash})
        if not key_doc:
            return None

        user_id = key_doc["user_id"]
        configs_col = _get_collection("user_configs")
        config_doc = configs_col.find_one({"user_id": user_id})

        if config_doc:
            config = {
                "user_id": user_id,
                "ner_model_name": config_doc.get("ner_model_name", "tabiya/roberta-base-job-ner"),
                "nel_model_name": config_doc.get("nel_model_name", "all-MiniLM-L6-v2"),
                "taxonomy_model_id": config_doc.get("taxonomy_model_id", "generic"),
                "ner_type": config_doc.get("ner_type", "SELF_HOSTED_LLM"),
                "nel_type": config_doc.get("nel_type", "generic"),
            }
        else:
            config = {"user_id": user_id, **_default_config()}

        _in_memory_cache[key_hash] = (config, time.time() + API_KEY_CACHE_TTL)
        return config

    except Exception as e:
        log.warning("MongoDB lookup failed, using default config: %s", e)
        return _default_config()


def _default_config() -> dict:
    return {
        "user_id": "default",
        "ner_model_name": "tabiya/roberta-base-job-ner",
        "nel_model_name": "all-MiniLM-L6-v2",
        "taxonomy_model_id": "generic",
        "ner_type": "SELF_HOSTED_LLM",
        "nel_type": "generic",
    }

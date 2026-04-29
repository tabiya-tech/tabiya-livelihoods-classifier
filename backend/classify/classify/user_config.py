"""User configuration and auth helpers.

Two auth paths:
  - API key (X-Api-Key header)  → classify/batch endpoints
    The API Gateway validates the key; the backend looks up user config from MongoDB.

  - Firebase Bearer token       → /v1/user/* dashboard endpoints
    The API Gateway verifies the Firebase JWT and forwards the decoded user info
    as a base64-encoded JSON string in the `x-apigateway-api-userinfo` header.
    In local development (TARGET_ENVIRONMENT_TYPE=local) the Bearer token is
    decoded directly without signature verification.
"""

import base64
import hashlib
import json
import logging
import time
from typing import Optional

from google.cloud import api_keys_v2
from google.cloud.api_keys_v2.types import Key, Restrictions, ApiTarget

_gcp_keys_async_client: api_keys_v2.ApiKeysAsyncClient | None = None


def _get_gcp_keys_client() -> api_keys_v2.ApiKeysAsyncClient:
    global _gcp_keys_async_client
    if _gcp_keys_async_client is None:
        _gcp_keys_async_client = api_keys_v2.ApiKeysAsyncClient()
    return _gcp_keys_async_client

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

from classify.config import (
    API_KEY_CACHE_TTL,
    GCP_PROJECT_ID,
    GCP_API_MANAGED_SERVICE,
    MONGODB_URI,
    TARGET_ENVIRONMENT_TYPE,
)
from classify.db import ApplicationDBProvider

log = logging.getLogger("classify-api")

_api_key_cache: dict = {}  # key_hash -> (user_config, expires_at)

_bearer_scheme = HTTPBearer(scheme_name="firebase", auto_error=False)
_api_key_header = APIKeyHeader(scheme_name="api_key", name="x-api-key", auto_error=False)


# ── MongoDB ────────────────────────────────────────────────────────────────

async def _get_collection(name: str):
    db = await ApplicationDBProvider.get_db()
    return db[name]


# ── API key auth ───────────────────────────────────────────────────────────

def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


async def get_user_config(api_key: str) -> Optional[dict]:
    """Resolve an API key to user config.

    Returns a dict with user_id + model config fields.
    Returns None if the key is unknown or revoked.
    Falls back to default config if MongoDB is not configured.
    """
    if not MONGODB_URI:
        return _default_config()

    key_hash = _hash_key(api_key)

    cached = _api_key_cache.get(key_hash)
    if cached:
        config, expires_at = cached
        if time.time() < expires_at:
            return config
        del _api_key_cache[key_hash]

    try:
        api_keys_col = await _get_collection("api_keys")
        key_doc = await api_keys_col.find_one(
            {"key_hash": key_hash, "revoked": {"$ne": True}}
        )
        if not key_doc:
            return None

        await api_keys_col.update_one(
            {"key_hash": key_hash},
            {"$set": {"last_used_at": time.time()}},
        )

        user_id = key_doc["user_id"]
        user_configs_col = await _get_collection("user_configs")
        config_doc = await user_configs_col.find_one({"user_id": user_id})
        config = _build_config(user_id, config_doc)
        _api_key_cache[key_hash] = (config, time.time() + API_KEY_CACHE_TTL)
        return config

    except Exception as e:
        log.warning("MongoDB lookup failed, using default config: %s", e)
        return _default_config()


async def get_api_key_user(api_key: Optional[str] = Depends(_api_key_header)) -> dict:
    """FastAPI dependency: returns user config for the request's API key.

    In production the gateway has already validated the key; we just do the
    MongoDB lookup to resolve it to a user config.

    Locally (TARGET_ENVIRONMENT_TYPE=local) the gateway is absent, so:
      - If a key is provided, look it up (allows testing with real keys).
      - If no key is provided, return the default config so the service is
        usable without any auth setup.
    """
    if TARGET_ENVIRONMENT_TYPE == "local":
        if not api_key:
            return _default_config()
        config = await get_user_config(api_key)
        return config if config is not None else _default_config()

    # Production: gateway has validated the key; enforce its presence and resolve.
    if not api_key:
        raise HTTPException(status_code=401, detail="x-api-key header required")
    config = await get_user_config(api_key)
    if config is None:
        # Key was valid at the gateway but revoked since — treat as 403.
        raise HTTPException(status_code=403, detail="API key revoked")
    return config


def _build_config(user_id: str, config_doc: Optional[dict]) -> dict:
    base = {"user_id": user_id}
    if config_doc:
        return {
            **base,
            "ner_model_name": config_doc.get("ner_model_name", "tabiya/roberta-base-job-ner"),
            "nel_model_name": config_doc.get("nel_model_name", "all-MiniLM-L6-v2"),
            "taxonomy_model_id": config_doc.get("taxonomy_model_id", "generic"),
            "ner_type": config_doc.get("ner_type", "SELF_HOSTED_LLM"),
            "nel_type": config_doc.get("nel_type", "generic"),
        }
    return {**base, **_default_config()}


def _default_config() -> dict:
    return {
        "user_id": "default",
        "ner_model_name": "tabiya/roberta-base-job-ner",
        "nel_model_name": "all-MiniLM-L6-v2",
        "taxonomy_model_id": "generic",
        "ner_type": "SELF_HOSTED_LLM",
        "nel_type": "generic",
    }


# ── Firebase / dashboard auth ──────────────────────────────────────────────

def _decode_gateway_user_info(auth_info_b64: str) -> dict:
    """Decode the base64 user info header forwarded by the API Gateway.

    The gateway verifies the Firebase JWT and encodes the claims as base64 JSON.
    Padding may be missing — handle all cases.
    """
    padding_needed = len(auth_info_b64) % 4
    if padding_needed == 1:
        raise ValueError("Invalid base64 input")
    elif padding_needed == 2:
        auth_info_b64 += "=="
    elif padding_needed == 3:
        auth_info_b64 += "="
    decoded = base64.b64decode(auth_info_b64.encode("utf-8"))
    return json.loads(decoded.decode("utf-8"))


def get_firebase_uid(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme)) -> str:
    """FastAPI dependency: returns the Firebase uid of the authenticated user.

    In production the API Gateway has already verified the token and placed the
    decoded claims in x-apigateway-api-userinfo. In local dev the Bearer token
    is decoded directly without signature verification.
    """
    try:
        if TARGET_ENVIRONMENT_TYPE != "local":
            auth_info_b64 = request.headers.get("x-apigateway-api-userinfo")
            if not auth_info_b64:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
            token_info = _decode_gateway_user_info(auth_info_b64)
        else:
            if not credentials:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
            token_info = jwt.decode(credentials.credentials, options={"verify_signature": False})

        uid = token_info.get("sub") or token_info.get("user_id")
        if not uid:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        return uid

    except HTTPException:
        raise
    except Exception as e:
        log.warning("Auth error: %s - %s", e.__class__.__name__, e)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


# ── User config CRUD ───────────────────────────────────────────────────────

async def get_user_config_for_uid(uid: str) -> dict:
    if not MONGODB_URI:
        return _default_config()
    col = await _get_collection("user_configs")
    doc = await col.find_one({"user_id": uid})
    return _build_config(uid, doc)


async def set_user_config_for_uid(uid: str, updates: dict) -> None:
    allowed = {"ner_type", "nel_type", "ner_model_name", "nel_model_name", "taxonomy_model_id"}
    safe = {k: v for k, v in updates.items() if k in allowed}
    col = await _get_collection("user_configs")
    await col.update_one(
        {"user_id": uid},
        {"$set": safe},
        upsert=True,
    )


# ── API key CRUD ───────────────────────────────────────────────────────────

async def list_api_keys_for_uid(uid: str) -> list:
    col = await _get_collection("api_keys")
    cursor = col.find(
        {"user_id": uid, "revoked": {"$ne": True}},
        {"_id": 0, "key_hash": 0},
    )
    return await cursor.to_list(length=None)


async def create_api_key_for_uid(uid: str, label: str) -> dict:
    now = time.time()

    client = _get_gcp_keys_client()
    key = Key(
        display_name=f"{label} ({uid})",
        restrictions=Restrictions(
            api_targets=[ApiTarget(service=GCP_API_MANAGED_SERVICE)],
        ),
    )
    op = await client.create_key(parent=f"projects/{GCP_PROJECT_ID}/locations/global", key=key)
    created = await op.result()

    # GCP key name format: projects/{project}/locations/global/keys/{key_id}
    gcp_key_id = created.name.split("/")[-1]
    plain_key = created.key_string

    doc = {
        "key_id": gcp_key_id,
        "gcp_key_name": created.name,
        "user_id": uid,
        "label": label,
        "key_hash": _hash_key(plain_key),
        "created_at": now,
        "last_used_at": None,
        "revoked": False,
    }
    col = await _get_collection("api_keys")
    await col.insert_one(doc)
    meta = {k: v for k, v in doc.items() if k not in ("_id", "key_hash", "gcp_key_name")}
    return {"key": plain_key, "meta": meta}


async def delete_api_key_for_uid(uid: str, key_id: str) -> bool:
    col = await _get_collection("api_keys")
    doc = await col.find_one({"key_id": key_id, "user_id": uid})
    if not doc:
        return False

    gcp_key_name = doc.get("gcp_key_name")
    if gcp_key_name:
        try:
            client = _get_gcp_keys_client()
            op = await client.delete_key(name=gcp_key_name)
            await op.result()
        except Exception as exc:
            log.warning("Failed to delete GCP API key %s: %s", gcp_key_name, exc)

    await col.update_one({"key_id": key_id, "user_id": uid}, {"$set": {"revoked": True}})
    return True


# ── Usage stats ────────────────────────────────────────────────────────────

async def get_usage_for_uid(uid: str) -> list:
    if not MONGODB_URI:
        return []
    cutoff = time.time() - 30 * 86400
    pipeline = [
        {"$match": {"user_id": uid, "ts": {"$gte": cutoff}}},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": {"$toDate": {"$multiply": ["$ts", 1000]}}}},
            "count": {"$sum": 1},
        }},
        {"$sort": {"_id": 1}},
        {"$project": {"_id": 0, "date": "$_id", "count": 1}},
    ]
    col = await _get_collection("usage_events")
    return await col.aggregate(pipeline).to_list(length=None)


async def record_usage_event(uid: str) -> None:
    if not MONGODB_URI or uid == "default":
        return
    try:
        col = await _get_collection("usage_events")
        await col.insert_one({"user_id": uid, "ts": time.time()})
    except Exception as e:
        log.warning("Failed to record usage event: %s", e)

"""Resolve an API key to its owning user id.

API-key requests arrive through the GCP gateway with a validated `x-api-key`
header but NO `x-apigateway-api-userinfo` (the gateway only forwards user
claims for Firebase). The api_keys collection stores, per key, the sha256
`key_hash` and the owning `user_id`. Hashing the incoming key the same way it
was hashed at creation and looking it up recovers the owner — so an API-key
caller resolves to the same uid (and therefore the same per-user model config)
as that user's Firebase session.

This mirrors the identical helper in classify-v2; both services read the same
`api_keys` collection layout in the shared application DB. classify-v2 owns
writes to that collection; nel-v2 only reads it here.
"""

import hashlib
import logging

from motor.motor_asyncio import AsyncIOMotorDatabase

_logger = logging.getLogger(__name__)

_API_KEYS_COLLECTION = "api_keys"


def hash_api_key(plaintext_key: str) -> str:
    """sha256-hex, matching how keys are hashed at creation time."""
    return hashlib.sha256(plaintext_key.encode("utf-8")).hexdigest()


async def resolve_user_id_from_api_key(
    application_db: AsyncIOMotorDatabase, plaintext_key: str
) -> str | None:
    """Return the user_id that owns this API key, or None if unknown/revoked."""
    key_hash = hash_api_key(plaintext_key)
    doc = await application_db[_API_KEYS_COLLECTION].find_one(
        {"key_hash": key_hash, "revoked": {"$ne": True}},
        {"user_id": 1, "_id": 0},
    )
    if not doc:
        return None
    return doc.get("user_id")

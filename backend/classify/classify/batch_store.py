"""Batch job state management backed by MongoDB."""

import time
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from classify.db import ApplicationDBProvider

COLLECTION = "batch_jobs"


async def _col():
    db = await ApplicationDBProvider.get_db()
    return db[COLLECTION]


async def ensure_indexes() -> None:
    """Create indexes once at startup. Safe to call multiple times (idempotent)."""
    col = await _col()
    # _id index covers primary batch_id lookups.
    # user_id index covers listing a user's batches.
    await col.create_index("user_id")


async def create_batch(batch_id: str, user_id: str, total: int) -> None:
    col = await _col()
    await col.insert_one({
        "_id": batch_id,
        "user_id": user_id,
        "status": "processing",
        "total": total,
        "processed": 0,
        "results": [],
        "created_at": time.time(),
        "completed_at": None,
    })


async def get_batch_for_user(batch_id: str, user_id: str) -> Optional[dict]:
    """Fetch a batch only if it belongs to the given user."""
    col = await _col()
    doc = await col.find_one({"_id": batch_id, "user_id": user_id})
    if doc is None:
        return None
    doc["batch_id"] = doc.pop("_id")
    return doc


async def update_batch(batch_id: str, result: dict) -> None:
    # Single atomic operation — no read-modify-write.
    col = await _col()
    await col.update_one(
        {"_id": batch_id},
        {
            "$push": {"results": result},
            "$inc": {"processed": 1},
        },
    )


async def complete_batch(batch_id: str) -> None:
    col = await _col()
    await col.update_one(
        {"_id": batch_id},
        {"$set": {"status": "completed", "completed_at": time.time()}},
    )


async def fail_batch(batch_id: str, error: str) -> None:
    col = await _col()
    await col.update_one(
        {"_id": batch_id},
        {"$set": {"status": "failed", "error": error, "completed_at": time.time()}},
    )

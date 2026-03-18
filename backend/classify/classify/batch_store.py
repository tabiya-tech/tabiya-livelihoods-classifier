"""Batch job state management backed by Redis (GCP Memorystore)."""

import json
import time
from typing import Optional

import redis.asyncio as aioredis

from classify.config import REDIS_URL, REDIS_BATCH_TTL

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


async def create_batch(batch_id: str, total: int) -> None:
    r = await get_redis()
    state = {
        "status": "processing",
        "total": total,
        "processed": 0,
        "results": [],
        "created_at": time.time(),
        "completed_at": None,
    }
    await r.setex(f"batch:{batch_id}", REDIS_BATCH_TTL, json.dumps(state))


async def get_batch(batch_id: str) -> Optional[dict]:
    r = await get_redis()
    raw = await r.get(f"batch:{batch_id}")
    if raw is None:
        return None
    return json.loads(raw)


async def update_batch(batch_id: str, processed: int, result: dict) -> None:
    r = await get_redis()
    raw = await r.get(f"batch:{batch_id}")
    if raw is None:
        return
    state = json.loads(raw)
    state["processed"] = processed
    state["results"].append(result)
    await r.setex(f"batch:{batch_id}", REDIS_BATCH_TTL, json.dumps(state))


async def complete_batch(batch_id: str) -> None:
    r = await get_redis()
    raw = await r.get(f"batch:{batch_id}")
    if raw is None:
        return
    state = json.loads(raw)
    state["status"] = "completed"
    state["completed_at"] = time.time()
    await r.setex(f"batch:{batch_id}", REDIS_BATCH_TTL, json.dumps(state))

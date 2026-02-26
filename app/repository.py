"""
Repository layer — abstracts database access so the classifier
never imports MongoDB (or any other DB) directly.

Usage:
    repo = MongoJobRepository(mongo_uri, db_name)
    job = await repo.get_job("abc123")

    # For testing without a database:
    repo = InMemoryJobRepository()
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime, timezone
import logging
import os

from dotenv import load_dotenv

from util.job_text import build_input_text, compute_hash

load_dotenv()

logger = logging.getLogger("repository")


# ─── Abstract Interface ───────────────────────────────────────────

class JobRepository(ABC):
    """
    Interface that the classifier depends on.
    Swap implementations to change databases without touching classifier code.
    """

    @abstractmethod
    async def get_job(self, fingerprint: str) -> Optional[Dict]:
        """Fetch a raw job by its fingerprint."""
        ...

    @abstractmethod
    async def get_unclassified_jobs(self, limit: int = 100, platform: Optional[str] = None) -> List[Dict]:
        """Fetch raw jobs that haven't been classified yet, optionally filtered by platform."""
        ...

    @abstractmethod
    async def save_classification(self, result: Dict) -> None:
        """Upsert a classified-jobs document."""
        ...

    @abstractmethod
    async def get_classification(self, fingerprint: str) -> Optional[Dict]:
        """Fetch a classification by job fingerprint."""
        ...

    @abstractmethod
    async def is_already_classified(self, fingerprint: str, input_text_hash: str) -> bool:
        """Check if a job has already been classified with the same text."""
        ...

    @abstractmethod
    async def get_all_classified_jobs(self, limit: int = 500, platform: Optional[str] = None) -> List[Dict]:
        """Fetch classified jobs, optionally filtered by source platform."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up connections."""
        ...


# ─── MongoDB Implementation ───────────────────────────────────────

class MongoJobRepository(JobRepository):
    """
    MongoDB-specific implementation. This is the only place that
    imports motor/pymongo. Replace this class to swap databases.
    """

    def __init__(self, mongo_uri: Optional[str] = None, db_name: Optional[str] = None):
        from motor.motor_asyncio import AsyncIOMotorClient

        # Also load .env from the scraper directory if classifier .env doesn't have MongoDB config
        scraper_env = os.path.join(os.path.dirname(__file__), "..", "..", "job_scraper", ".env")
        if os.path.exists(scraper_env):
            load_dotenv(scraper_env, override=False)

        uri = mongo_uri or os.getenv("APPLICATION_MONGODB_URI") or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        name = db_name or os.getenv("APPLICATION_DATABASE_NAME", "horizon-scraper-dev")

        self.client = AsyncIOMotorClient(uri)
        self.db = self.client[name]
        self.raw_jobs = self.db["raw-jobs"]
        self.classified_jobs = self.db["classified-jobs"]

    async def get_job(self, fingerprint: str) -> Optional[Dict]:
        return await self.raw_jobs.find_one({"job_fingerprint": fingerprint})

    async def get_unclassified_jobs(self, limit: int = 100, platform: Optional[str] = None) -> List[Dict]:
        match_stage: Dict = {"classification": {"$size": 0}}
        if platform:
            match_stage["sources.platform"] = platform

        pipeline = [
            {
                "$lookup": {
                    "from": "classified-jobs",
                    "localField": "job_fingerprint",
                    "foreignField": "job_fingerprint",
                    "as": "classification",
                }
            },
            {"$match": match_stage},
            {"$project": {"classification": 0}},
            {"$sort": {"created_at": -1}},
            {"$limit": limit},
        ]
        cursor = self.raw_jobs.aggregate(pipeline)
        return await cursor.to_list(length=limit)

    async def save_classification(self, result: Dict) -> None:
        now = datetime.now(timezone.utc)
        result["updated_at"] = now
        set_doc = {k: v for k, v in result.items() if k != "created_at"}

        await self.classified_jobs.update_one(
            {"job_fingerprint": result["job_fingerprint"]},
            {"$set": set_doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )

    async def get_classification(self, fingerprint: str) -> Optional[Dict]:
        return await self.classified_jobs.find_one({"job_fingerprint": fingerprint})

    async def is_already_classified(self, fingerprint: str, input_text_hash: str) -> bool:
        doc = await self.classified_jobs.find_one({
            "job_fingerprint": fingerprint,
            "$or": [
                {"input_text_hash": input_text_hash},
                {"metadata.input_text_hash": input_text_hash},
            ],
        })
        return doc is not None

    async def get_all_classified_jobs(self, limit: int = 500, platform: Optional[str] = None) -> List[Dict]:
        """Fetch classified jobs, optionally filtered by source platform."""
        query: Dict = {}
        if platform:
            query["source_fields.source_platform"] = platform
        cursor = self.classified_jobs.find(query).sort("classified_at", -1).limit(limit)
        return await cursor.to_list(length=limit)

    async def watch_raw_jobs(self):
        """Yield full documents from the raw-jobs change stream (MongoDB-specific)."""
        pipeline = [
            {"$match": {"operationType": {"$in": ["insert", "replace", "update"]}}}
        ]
        async with self.raw_jobs.watch(pipeline, full_document="updateLookup") as stream:
            async for change in stream:
                doc = change.get("fullDocument")
                if doc:
                    yield doc

    async def close(self) -> None:
        self.client.close()


# ─── In-Memory Implementation (for testing) ───────────────────────

class InMemoryJobRepository(JobRepository):
    """
    Stores everything in Python dicts. No database needed.
    Useful for unit tests and local development without credentials.
    """

    def __init__(self):
        self.raw_jobs: Dict[str, Dict] = {}
        self.classifications: Dict[str, Dict] = {}

    async def get_job(self, fingerprint: str) -> Optional[Dict]:
        return self.raw_jobs.get(fingerprint)

    async def get_unclassified_jobs(self, limit: int = 100) -> List[Dict]:
        unclassified = [
            job for fp, job in self.raw_jobs.items()
            if fp not in self.classifications
        ]
        return unclassified[:limit]

    async def save_classification(self, result: Dict) -> None:
        now = datetime.now(timezone.utc)
        result.setdefault("created_at", now)
        result["updated_at"] = now
        self.classifications[result["job_fingerprint"]] = result

    async def get_classification(self, fingerprint: str) -> Optional[Dict]:
        return self.classifications.get(fingerprint)

    async def is_already_classified(self, fingerprint: str, input_text_hash: str) -> bool:
        doc = self.classifications.get(fingerprint)
        if not doc:
            return False
        return doc.get("input_text_hash") == input_text_hash

    async def get_all_classified_jobs(self, limit: int = 500, platform: Optional[str] = None) -> List[Dict]:
        results = list(self.classifications.values())
        if platform:
            results = [r for r in results if r.get("source_fields", {}).get("source_platform") == platform]
        return results[:limit]

    def add_raw_job(self, job: Dict) -> None:
        """Helper to seed test data."""
        self.raw_jobs[job["job_fingerprint"]] = job

    async def close(self) -> None:
        pass


# ─── Helper ───────────────────────────────────────────────────────

def compute_input_text_hash(title: str, description: str) -> str:
    """Compute SHA256 of the classifier input text for deduplication."""
    text = build_input_text({"title": title, "description": description})
    return compute_hash(text or "")

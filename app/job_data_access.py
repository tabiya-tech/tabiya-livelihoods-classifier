from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime, timezone
import logging
import os

from dotenv import load_dotenv

from util.job_text import build_input_text, compute_hash

load_dotenv()

logger = logging.getLogger("repository")

APPLICATION_MONGODB_URI = os.getenv("APPLICATION_MONGODB_URI") or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
APPLICATION_DATABASE_NAME = os.getenv("APPLICATION_DATABASE_NAME", "horizon-scraper-dev")


#  Abstract Interface 

class JobRepository(ABC):
    """Interface that the classifier depends on.
    Swap implementations to change databases without touching classifier code.
    """

    @abstractmethod
    def get_job(self, fingerprint: str) -> Optional[Dict]:
        ...

    @abstractmethod
    def get_unclassified_jobs(self, limit: int = 100, platform: Optional[str] = None) -> List[Dict]:
        ...

    @abstractmethod
    def save_classification(self, result: Dict) -> None:
        ...

    @abstractmethod
    def get_classification(self, fingerprint: str) -> Optional[Dict]:
        ...

    @abstractmethod
    def is_already_classified(self, fingerprint: str, input_text_hash: str) -> bool:
        ...

    @abstractmethod
    def get_all_classified_jobs(self, limit: int = 500, platform: Optional[str] = None) -> List[Dict]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


# MongoDB Implementation (sync pymongo) 

class MongoJobRepository(JobRepository):
    """Synchronous MongoDB implementation using pymongo."""

    def __init__(self, mongo_uri: Optional[str] = None, db_name: Optional[str] = None):
        from pymongo import MongoClient

        scraper_env = os.path.join(os.path.dirname(__file__), "..", "job_scraper", ".env")
        if os.path.exists(scraper_env):
            load_dotenv(scraper_env, override=False)

        uri = mongo_uri or APPLICATION_MONGODB_URI
        name = db_name or APPLICATION_DATABASE_NAME

        self.client = MongoClient(uri)
        self.db = self.client[name]
        self.raw_jobs = self.db["raw-jobs"]
        self.classified_jobs = self.db["classified-jobs"]

    def get_job(self, fingerprint: str) -> Optional[Dict]:
        return self.raw_jobs.find_one({"job_fingerprint": fingerprint})

    def get_unclassified_jobs(self, limit: int = 100, platform: Optional[str] = None) -> List[Dict]:
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
        return list(self.raw_jobs.aggregate(pipeline))

    def save_classification(self, result: Dict) -> None:
        now = datetime.now(timezone.utc)
        result["updated_at"] = now
        set_doc = {k: v for k, v in result.items() if k != "created_at"}

        self.classified_jobs.update_one(
            {"job_fingerprint": result["job_fingerprint"]},
            {"$set": set_doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )

    def get_classification(self, fingerprint: str) -> Optional[Dict]:
        return self.classified_jobs.find_one({"job_fingerprint": fingerprint})

    def is_already_classified(self, fingerprint: str, input_text_hash: str) -> bool:
        doc = self.classified_jobs.find_one({
            "job_fingerprint": fingerprint,
            "$or": [
                {"input_text_hash": input_text_hash},
                {"metadata.input_text_hash": input_text_hash},
            ],
        })
        return doc is not None

    def get_all_classified_jobs(self, limit: int = 500, platform: Optional[str] = None) -> List[Dict]:
        query: Dict = {}
        if platform:
            query["source_fields.source_platform"] = platform
        cursor = self.classified_jobs.find(query).sort("classified_at", -1).limit(limit)
        return list(cursor)

    def watch_raw_jobs(self):
        """Yield full documents from the raw-jobs change stream."""
        pipeline = [
            {"$match": {"operationType": {"$in": ["insert", "replace", "update"]}}}
        ]
        with self.raw_jobs.watch(pipeline, full_document="updateLookup") as stream:
            for change in stream:
                doc = change.get("fullDocument")
                if doc:
                    yield doc

    def close(self) -> None:
        self.client.close()


# In-Memory Implementation (for testing)

class InMemoryJobRepository(JobRepository):
    """Stores everything in Python dicts.  No database needed.
    Useful for unit tests and local development without credentials.
    """

    def __init__(self):
        self.raw_jobs: Dict[str, Dict] = {}
        self.classifications: Dict[str, Dict] = {}

    def get_job(self, fingerprint: str) -> Optional[Dict]:
        return self.raw_jobs.get(fingerprint)

    def get_unclassified_jobs(self, limit: int = 100, platform: Optional[str] = None) -> List[Dict]:
        unclassified = [
            job for fp, job in self.raw_jobs.items()
            if fp not in self.classifications
        ]
        return unclassified[:limit]

    def save_classification(self, result: Dict) -> None:
        now = datetime.now(timezone.utc)
        result.setdefault("created_at", now)
        result["updated_at"] = now
        self.classifications[result["job_fingerprint"]] = result

    def get_classification(self, fingerprint: str) -> Optional[Dict]:
        return self.classifications.get(fingerprint)

    def is_already_classified(self, fingerprint: str, input_text_hash: str) -> bool:
        doc = self.classifications.get(fingerprint)
        if not doc:
            return False
        return doc.get("input_text_hash") == input_text_hash

    def get_all_classified_jobs(self, limit: int = 500, platform: Optional[str] = None) -> List[Dict]:
        results = list(self.classifications.values())
        if platform:
            results = [r for r in results if r.get("source_fields", {}).get("source_platform") == platform]
        return results[:limit]

    def add_raw_job(self, job: Dict) -> None:
        """Helper to seed test data."""
        self.raw_jobs[job["job_fingerprint"]] = job

    def close(self) -> None:
        pass


# Helper

def compute_input_text_hash(title: str, description: str) -> str:
    """Compute SHA256 of the classifier input text for deduplication."""
    text = build_input_text({"title": title, "description": description})
    return compute_hash(text or "")

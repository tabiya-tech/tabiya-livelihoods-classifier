"""
Change Stream Worker — watches MongoDB raw-jobs for new/updated documents
and sends them to the Classify API automatically.

All database access goes through the JobRepository interface. The worker
never imports motor/pymongo directly.

Usage:
    python app/worker/change_stream_worker.py
    python app/worker/change_stream_worker.py --dry-run   # logs only, no writes

Environment:
    CLASSIFY_API_URL  — default http://localhost:5001
    APPLICATION_MONGODB_URI / MONGODB_URI — MongoDB connection string
    APPLICATION_DATABASE_NAME — default horizon-scraper-dev
"""

import sys
import os
import time
import asyncio
import logging
import argparse
from typing import Optional
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv

from app.server.common import setup_logging
from util.job_text import build_input_text, compute_hash

load_dotenv()
scraper_env = os.path.join(os.path.dirname(__file__), "..", "..", "..", "job_scraper", ".env")
if os.path.exists(scraper_env):
    load_dotenv(scraper_env, override=False)

setup_logging()
log = logging.getLogger("change-stream-worker")

CLASSIFY_API_URL = os.getenv("CLASSIFY_API_URL", "http://localhost:5001")

ALL_SOURCE_FIELDS = [
    "title", "employer", "location", "description",
    "employment_type", "education", "experience",
    "salary_text", "posted_date", "closing_date",
    "application_url", "source_platform",
]


def extract_source_fields(doc: dict) -> dict:
    """Pull all known source fields from a raw-jobs document."""
    fields = {}
    for key in ALL_SOURCE_FIELDS:
        val = doc.get(key)
        if val is not None:
            fields[key] = val
    # Map salary_text → salary and closing_date → expiry_date for downstream consumers
    if "salary_text" in fields:
        fields["salary"] = fields.pop("salary_text")
    if "closing_date" in fields:
        fields["expiry_date"] = fields.pop("closing_date")
    return fields


def classify_job(input_text: str, job_fingerprint: str, dry_run: bool = False) -> Optional[dict]:
    """Send a job to the Classify API. Returns the response dict or None on failure."""
    if dry_run:
        log.info(f"  [DRY-RUN] Would classify job {job_fingerprint}: {input_text[:80]}...")
        return {"dry_run": True}

    try:
        resp = requests.post(
            f"{CLASSIFY_API_URL}/v1/classify",
            json={"text": input_text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        log.error(f"  Classify API error for {job_fingerprint}: {e}")
        return None


async def watch_change_stream(dry_run: bool = False):
    """Connect to MongoDB via JobRepository and watch raw-jobs for changes."""
    from app.repository import MongoJobRepository

    repo = MongoJobRepository()
    log.info("Watching raw-jobs change stream via JobRepository... (Ctrl+C to stop)")

    try:
        async for doc in repo.watch_raw_jobs():
            fingerprint = doc.get("job_fingerprint", "unknown")
            input_text = build_input_text(doc)

            if not input_text:
                log.warning(f"  Skipping {fingerprint} — no title or description")
                continue

            input_hash = compute_hash(input_text)

            if await repo.is_already_classified(fingerprint, input_hash):
                log.info(f"  Skipping {fingerprint} — already classified (hash match)")
                continue

            log.info(f"  New job detected: {fingerprint}")
            result = classify_job(input_text, fingerprint, dry_run=dry_run)

            if result and not dry_run:
                classification_doc = {
                    "job_fingerprint": fingerprint,
                    "input_text_hash": input_hash,
                    "classification": result.get("classification", {}),
                    "metadata": result.get("metadata", {}),
                    "source_fields": extract_source_fields(doc),
                    "classified_at": time.time(),
                    "status": "completed",
                }
                await repo.save_classification(classification_doc)
                log.info(f"  Saved classification for {fingerprint}")
            elif result:
                log.info(f"  [DRY-RUN] Classification complete for {fingerprint}")
    finally:
        await repo.close()


async def backfill_unclassified(dry_run: bool = False, platform: Optional[str] = None):
    """Batch-process any raw-jobs that haven't been classified yet."""
    from app.repository import MongoJobRepository

    repo = MongoJobRepository()
    log.info(f"Backfilling unclassified jobs (platform={platform or 'all'})...")

    try:
        jobs = await repo.get_unclassified_jobs(limit=500, platform=platform)
        log.info(f"Found {len(jobs)} unclassified jobs")

        for i, doc in enumerate(jobs, 1):
            fingerprint = doc.get("job_fingerprint", "unknown")
            input_text = build_input_text(doc)

            if not input_text:
                log.warning(f"  [{i}/{len(jobs)}] Skipping {fingerprint} — no text")
                continue

            input_hash = compute_hash(input_text)

            if await repo.is_already_classified(fingerprint, input_hash):
                log.info(f"  [{i}/{len(jobs)}] Skipping {fingerprint} — already classified")
                continue

            log.info(f"  [{i}/{len(jobs)}] Classifying: {fingerprint}")
            result = classify_job(input_text, fingerprint, dry_run=dry_run)

            if result and not dry_run:
                classification_doc = {
                    "job_fingerprint": fingerprint,
                    "input_text_hash": input_hash,
                    "classification": result.get("classification", {}),
                    "metadata": result.get("metadata", {}),
                    "source_fields": extract_source_fields(doc),
                    "classified_at": time.time(),
                    "status": "completed",
                }
                await repo.save_classification(classification_doc)
                log.info(f"  Saved classification for {fingerprint}")
            elif result:
                log.info(f"  [{i}/{len(jobs)}] [DRY-RUN] done")

            time.sleep(0.2)

        log.info("Backfill complete")
    finally:
        await repo.close()


MAX_RETRIES = int(os.getenv("WORKER_MAX_RETRIES", "0"))
RETRY_DELAY = int(os.getenv("WORKER_RETRY_DELAY_SECONDS", "10"))


def main():
    parser = argparse.ArgumentParser(description="Change Stream Worker")
    parser.add_argument("--dry-run", action="store_true", help="Log only, don't write to DB")
    parser.add_argument("--backfill", action="store_true", help="Batch-classify existing unclassified jobs then exit")
    parser.add_argument("--platform", type=str, help="Filter backfill to a specific platform (e.g. gozambiajobs)")
    args = parser.parse_args()

    if args.dry_run:
        log.info("Running in DRY-RUN mode — no writes to MongoDB")

    if args.backfill:
        asyncio.run(backfill_unclassified(dry_run=args.dry_run, platform=args.platform))
        return

    retries = 0
    while True:
        try:
            asyncio.run(watch_change_stream(dry_run=args.dry_run))
            break
        except KeyboardInterrupt:
            log.info("Worker stopped by user.")
            break
        except Exception as e:
            retries += 1
            if MAX_RETRIES > 0 and retries > MAX_RETRIES:
                log.error(f"Max retries ({MAX_RETRIES}) exceeded. Exiting.")
                break
            log.error(f"Worker error (attempt {retries}): {e}")
            log.info(f"Reconnecting in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)


if __name__ == "__main__":
    main()

import sys
import os
import time
import logging
import argparse
from typing import Optional

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv

from app.server.common import setup_logging
from util.classified_job_schema import SCRAPER_ENRICHMENT_FIELD_NAMES
from util.job_text import build_input_text, compute_hash

load_dotenv()
scraper_env = os.path.join(os.path.dirname(__file__), "..", "..", "..", "job_scraper", ".env")
if os.path.exists(scraper_env):
    load_dotenv(scraper_env, override=False)

setup_logging()
log = logging.getLogger("change-stream-worker")

CLASSIFY_API_URL = os.getenv("CLASSIFY_API_URL", "http://localhost:5001")
MAX_RETRIES = int(os.getenv("WORKER_MAX_RETRIES", "0"))
RETRY_DELAY = int(os.getenv("WORKER_RETRY_DELAY_SECONDS", "10"))

ALL_SOURCE_FIELDS = [
    "title", "employer", "location", "description",
    "employment_type", "education", "experience",
    "salary_text", "posted_date", "closing_date",
    "application_url", "source_platform",
    *SCRAPER_ENRICHMENT_FIELD_NAMES,
]


def extract_source_fields(doc: dict) -> dict:
    """Pull all known source fields from a raw-jobs document."""
    fields = {}
    for key in ALL_SOURCE_FIELDS:
        val = doc.get(key)
        if val is not None:
            fields[key] = val
    if "salary_text" in fields:
        fields["salary"] = fields.pop("salary_text")
    if "closing_date" in fields:
        fields["expiry_date"] = fields.pop("closing_date")
    return fields


def classify_job(input_text: str, job_fingerprint: str, dry_run: bool = False) -> Optional[dict]:
    """Send a job to the Classify API.  Returns the response dict or None on failure."""
    if dry_run:
        log.info("  [DRY-RUN] Would classify job %s: %s...", job_fingerprint, input_text[:80])
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
        log.error("  Classify API error for %s: %s", job_fingerprint, e)
        return None


def watch_change_stream(dry_run: bool = False):
    """Connect to MongoDB via JobRepository and watch raw-jobs for changes."""
    from app.job_data_access import MongoJobRepository

    repo = MongoJobRepository()
    log.info("Watching raw-jobs change stream... (Ctrl+C to stop)")

    try:
        for doc in repo.watch_raw_jobs():
            fingerprint = doc.get("job_fingerprint", "unknown")
            input_text = build_input_text(doc)

            if not input_text:
                log.warning("  Skipping %s — no title or description", fingerprint)
                continue

            input_hash = compute_hash(input_text)

            if repo.is_already_classified(fingerprint, input_hash):
                log.info("  Skipping %s — already classified (hash match)", fingerprint)
                continue

            log.info("  New job detected: %s", fingerprint)
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
                repo.save_classification(classification_doc)
                log.info("  Saved classification for %s", fingerprint)
            elif result:
                log.info("  [DRY-RUN] Classification complete for %s", fingerprint)
    finally:
        repo.close()


def backfill_unclassified(dry_run: bool = False, platform: Optional[str] = None):
    """Batch-process any raw-jobs that haven't been classified yet."""
    from app.job_data_access import MongoJobRepository

    repo = MongoJobRepository()
    log.info("Backfilling unclassified jobs (platform=%s)...", platform or "all")

    try:
        jobs = repo.get_unclassified_jobs(limit=500, platform=platform)
        log.info("Found %d unclassified jobs", len(jobs))

        for i, doc in enumerate(jobs, 1):
            fingerprint = doc.get("job_fingerprint", "unknown")
            input_text = build_input_text(doc)

            if not input_text:
                log.warning("  [%d/%d] Skipping %s — no text", i, len(jobs), fingerprint)
                continue

            input_hash = compute_hash(input_text)

            if repo.is_already_classified(fingerprint, input_hash):
                log.info("  [%d/%d] Skipping %s — already classified", i, len(jobs), fingerprint)
                continue

            log.info("  [%d/%d] Classifying: %s", i, len(jobs), fingerprint)
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
                repo.save_classification(classification_doc)
                log.info("  Saved classification for %s", fingerprint)
            elif result:
                log.info("  [%d/%d] [DRY-RUN] done", i, len(jobs))

            time.sleep(0.2)

        log.info("Backfill complete")
    finally:
        repo.close()


def main():
    parser = argparse.ArgumentParser(description="Change Stream Worker")
    parser.add_argument("--dry-run", action="store_true", help="Log only, don't write to DB")
    parser.add_argument("--backfill", action="store_true", help="Batch-classify unclassified jobs then exit")
    parser.add_argument("--platform", type=str, help="Filter backfill to a specific platform")
    args = parser.parse_args()

    if args.dry_run:
        log.info("Running in DRY-RUN mode — no writes to MongoDB")

    if args.backfill:
        backfill_unclassified(dry_run=args.dry_run, platform=args.platform)
        return

    retries = 0
    while True:
        try:
            watch_change_stream(dry_run=args.dry_run)
            break
        except KeyboardInterrupt:
            log.info("Worker stopped by user.")
            break
        except Exception as e:
            retries += 1
            if MAX_RETRIES > 0 and retries > MAX_RETRIES:
                log.error("Max retries (%d) exceeded. Exiting.", MAX_RETRIES)
                break
            log.error("Worker error (attempt %d): %s", retries, e)
            log.info("Reconnecting in %ds...", RETRY_DELAY)
            time.sleep(RETRY_DELAY)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
End-to-end classifier job: reads ``scraped_jobs`` from MongoDB, calls Classify API,
writes results to ``classified_jobs``. Works for both Zambia and Kenya via TARGET_COUNTRY.

Scraper-style run-state is supported:
  - run logs collection: ``classifier_logs`` (open/in_progress -> completed/failed/stopped)
  - incremental state collection: ``classifier_state`` (last_cutoff_date per country)
"""
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Load .env before reading config
try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parent
    load_dotenv(_root / ".env")
except ImportError:
    pass

import requests
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from util.classified_job_schema import build_classified_job_document
from util.job_text import build_batch_classifier_input_text as build_input_text


def _get_config():
    """Load config from env based on TARGET_COUNTRY. No hardcoded secrets."""
    target = (os.getenv("TARGET_COUNTRY") or "zambia").strip().lower()
    if target not in ("kenya", "zambia", "south_africa", "argentina"):
        target = "zambia"

    prefix = f"{target.upper()}_"
    uri = (
        os.getenv(f"{prefix}MONGODB_URI")
        or os.getenv("APPLICATION_MONGODB_URI")
        or os.getenv("MONGODB_URI")
    )
    # Cloud Run often mounts only ZAMBIA_/KENYA_ secrets; SA / Argentina use the same cluster.
    if not uri and target in ("south_africa", "argentina"):
        uri = (
            os.getenv("ZAMBIA_MONGODB_URI")
            or os.getenv("KENYA_MONGODB_URI")
            or os.getenv("APPLICATION_MONGODB_URI")
            or os.getenv("MONGODB_URI")
        )
    if uri:
        uri = uri.strip().strip('"\'')
    if not uri:
        raise ValueError(
            f"MongoDB URI not set. Configure {prefix}MONGODB_URI, APPLICATION_MONGODB_URI, "
            "or MONGODB_URI via environment variables (for south_africa, ZAMBIA_MONGODB_URI or "
            "KENYA_MONGODB_URI if they reference the same cluster)."
        )

    if target == "zambia":
        _default_db = "ZambiaJobs_V2"
    elif target == "argentina":
        _default_db = "ArgentinaJobs_V2"
    else:
        _default_db = "KenyaJobs"
    db_name = (
        os.getenv(f"{prefix}DATABASE_NAME")
        or os.getenv("APPLICATION_DATABASE_NAME")
        or _default_db
    )
    src_coll = os.getenv("SOURCE_COLLECTION", "scraped_jobs_V2")
    tgt_coll = os.getenv("TARGET_COLLECTION", "hamz_classified_jobs")
    err_coll = os.getenv("ERRORS_COLLECTION", "hamz_classifier_errors")
    logs_coll = os.getenv("CLASSIFIER_LOGS_COLLECTION", "hamz_classifier_logs")
    state_coll = os.getenv("CLASSIFIER_STATE_COLLECTION", "hamz_classifier_state")

    classify_url = (os.getenv("CLASSIFY_API_URL") or "").strip()
    if not classify_url:
        raise ValueError("CLASSIFY_API_URL is required.")

    max_jobs_str = os.getenv("MAX_JOBS", "").strip()
    max_jobs = None
    if max_jobs_str:
        try:
            max_jobs = int(max_jobs_str)
        except ValueError:
            pass

    return {
        "target_country": target,
        "env_prefix": prefix.rstrip("_"),
        "mongo_uri": uri,
        "database_name": db_name,
        "source_collection": src_coll,
        "target_collection": tgt_coll,
        "errors_collection": err_coll,
        "logs_collection": logs_coll,
        "state_collection": state_coll,
        "classify_api_url": classify_url,
        "max_jobs": max_jobs,
    }


def get_mongo_client(uri: str) -> MongoClient:
    mongo_kwargs: dict = {}
    try:
        import certifi  # type: ignore
        mongo_kwargs["tlsCAFile"] = certifi.where()
    except Exception:
        pass
    return MongoClient(uri, **mongo_kwargs)


def call_classify_api(url: str, text: str, top_k: int = 3, min_similarity: float = 0.5) -> dict:
    """Call the Classify API and return the parsed JSON."""
    classify_endpoint = url if url.rstrip("/").endswith("/v1/classify") else f"{url.rstrip('/')}/v1/classify"
    payload = {
        "text": text,
        "options": {"top_k": top_k, "min_similarity": min_similarity},
    }
    resp = requests.post(classify_endpoint, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _cutoff_str_from_stored_last(last_cutoff) -> str | None:
    """Normalize ``last_cutoff`` from Mongo to ``YYYY-MM-DD`` (used as-is, no day overlap)."""
    try:
        if isinstance(last_cutoff, datetime):
            dt = (
                last_cutoff
                if last_cutoff.tzinfo
                else last_cutoff.replace(tzinfo=timezone.utc)
            )
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
        if isinstance(last_cutoff, str):
            return datetime.fromisoformat(
                str(last_cutoff).strip().replace("Z", "+00:00")
            ).strftime("%Y-%m-%d")
        return None
    except (TypeError, ValueError) as e:
        print(f"[WARN] Could not parse last_cutoff_date={last_cutoff!r}: {e}", file=sys.stderr)
        return None


def _resolve_classifier_cutoff(config: dict, db) -> str | None:
    """Scraper-like cutoff resolution for classifier incremental runs.

    Priority:
      1) Explicit env override: <COUNTRY>_CLASSIFIER_CUTOFF_DATE or CLASSIFIER_CUTOFF_DATE
      2) ``classifier_state``.last_cutoff_date (exact calendar date)
      3) None (process all)
    """
    try:
        prefix = config["env_prefix"]
        env_specific = f"{prefix}_CLASSIFIER_CUTOFF_DATE" if prefix else "CLASSIFIER_CUTOFF_DATE"
        env_override = (os.getenv(env_specific) or os.getenv("CLASSIFIER_CUTOFF_DATE") or "").strip()
        if env_override:
            print(
                f"[{config['target_country'].upper()}] Cutoff: env override ({env_specific}={env_override})"
            )
            return env_override

        state_doc = db[config["state_collection"]].find_one({"country": config["target_country"]}) or {}
        last_cutoff = state_doc.get("last_cutoff_date")
        if last_cutoff:
            cutoff_str = _cutoff_str_from_stored_last(last_cutoff)
            if cutoff_str:
                print(
                    f"[{config['target_country'].upper()}] Cutoff: incremental from state "
                    f"last_cutoff_date={cutoff_str} (exact)"
                )
                return cutoff_str

        print(f"[{config['target_country'].upper()}] Cutoff: no previous state (full scan)")
        return None
    except Exception as e:
        print(
            f"[{config['target_country'].upper()}] Cutoff resolution failed ({e}); using full scan",
            file=sys.stderr,
        )
        return None


def _open_classifier_log(config: dict, db, cutoff_date: str | None) -> str:
    now = datetime.now(timezone.utc)
    doc = {
        "country": config["target_country"],
        "source_collection": config["source_collection"],
        "target_collection": config["target_collection"],
        "cutoff_date": cutoff_date,
        "started_at": now,
        "updated_at": now,
        "status": "in_progress",
        "success": None,
        "progress": {"processed": 0, "success": 0, "failures": 0},
    }
    result = db[config["logs_collection"]].insert_one(doc)
    return str(result.inserted_id)


def _update_classifier_log(config: dict, db, log_id: str, processed: int, success: int, failures: int) -> None:
    db[config["logs_collection"]].update_one(
        {"_id": ObjectId(log_id)},
        {"$set": {
            "updated_at": datetime.now(timezone.utc),
            "progress.processed": int(processed),
            "progress.success": int(success),
            "progress.failures": int(failures),
        }},
    )


def _close_classifier_log(
    config: dict,
    db,
    log_id: str,
    processed: int,
    success: int,
    failures: int,
    *,
    ok: bool,
    error_message: str | None = None,
) -> None:
    status = "completed" if ok else "failed"
    db[config["logs_collection"]].update_one(
        {"_id": ObjectId(log_id)},
        {"$set": {
            "completed_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "status": status,
            "success": bool(ok),
            "error_message": error_message,
            "stats": {"processed": int(processed), "success": int(success), "failures": int(failures)},
        }},
    )


def _set_classifier_state(config: dict, db, jobs_processed: int) -> None:
    """Persist per-country state only on successful completion (scraper-style)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    db[config["state_collection"]].update_one(
        {"country": config["target_country"]},
        {"$set": {
            "country": config["target_country"],
            "last_cutoff_date": today,
            "last_run_at": datetime.now(timezone.utc),
            "jobs_processed": int(jobs_processed),
        }},
        upsert=True,
    )


def process_classification(config: dict):
    client = get_mongo_client(config["mongo_uri"])
    db = client[config["database_name"]]
    src_col = db[config["source_collection"]]
    tgt_col = db[config["target_collection"]]
    err_col = db[config["errors_collection"]]
    api_url = config["classify_api_url"]
    max_jobs = config["max_jobs"]

    log_id = _open_classifier_log(config, db, None)
    cutoff_date = _resolve_classifier_cutoff(config, db)

    db[config["logs_collection"]].update_one(
        {"_id": ObjectId(log_id)},
        {"$set": {"cutoff_date": cutoff_date, "updated_at": datetime.now(timezone.utc)}},
    )

    query = {}
    if cutoff_date:
        try:
            cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            # Anchor on immutable `created_at` only (set by scraper via $ifNull on first insert).
            # Avoids matching docs whose `updated_at` / `last_checked_at` get bumped on every scrape.
            query = {"created_at": {"$gte": cutoff_dt}}
        except ValueError:
            print(
                f"[{config['target_country'].upper()}] Invalid cutoff_date={cutoff_date}; falling back to full scan",
                file=sys.stderr,
            )
    cursor = src_col.find(query, no_cursor_timeout=True)
    processed = success = failures = 0

    print(f"[{config['target_country'].upper()}] Classifying from {config['source_collection']} -> {config['target_collection']}")
    if query:
        print(f"[{config['target_country'].upper()}] Incremental query active from cutoff {cutoff_date}")

    try:
        for job in cursor:
            if max_jobs is not None and processed >= max_jobs:
                break

            processed += 1
            job_id = job.get("_id")

            try:
                input_text = build_input_text(job)
                if not input_text:
                    print(f"[SKIP] Job {job_id} has no valid text to classify")
                    continue

                classify_result = call_classify_api(api_url, input_text)
                classification = classify_result.get("classification", {})
                metadata = classify_result.get("metadata", {})

                now_utc = datetime.now(timezone.utc)
                classified_doc = build_classified_job_document(
                    job,
                    job_id=job_id,
                    classification=classification,
                    metadata=metadata,
                    now_utc=now_utc,
                )

                # `classified_at` / `created_at` are immutable anchors: written only on the
                # first insert so the reranker's incremental filter on `classified_at` is stable.
                tgt_col.update_one(
                    {"job_id": job_id},
                    {
                        "$set": classified_doc,
                        "$setOnInsert": {
                            "classified_at": now_utc,
                            "created_at": now_utc,
                        },
                    },
                    upsert=True,
                )

                success += 1
                entity_counts = classification.get("entity_counts", {})
                print(f"[OK] Job {job_id} -> {entity_counts}")

            except (requests.RequestException, PyMongoError, Exception) as e:
                failures += 1
                tb = traceback.format_exc()
                print(f"[ERR] Job {job_id}: {e}", file=sys.stderr)
                err_col.insert_one({
                    "job_id": job_id,
                    "error": str(e),
                    "traceback": tb,
                    "api_url": api_url,
                    "timestamp": datetime.utcnow(),
                })

            # Live progress update every 25 documents.
            if processed % 25 == 0:
                _update_classifier_log(config, db, log_id, processed, success, failures)

        _update_classifier_log(config, db, log_id, processed, success, failures)
        _close_classifier_log(config, db, log_id, processed, success, failures, ok=True)
        _set_classifier_state(config, db, jobs_processed=processed)

    except Exception as e:
        _close_classifier_log(
            config,
            db,
            log_id,
            processed,
            success,
            failures,
            ok=False,
            error_message=str(e),
        )
        raise
    finally:
        cursor.close()
        client.close()

    print(f"Done. processed={processed}, success={success}, failures={failures}", file=sys.stderr)
    return processed, success, failures


def main():
    config = _get_config()
    print(f"TARGET_COUNTRY={config['target_country']}")
    print(f"Database: {config['database_name']}")
    print(f"Source: {config['source_collection']} -> Target: {config['target_collection']}")
    print(f"Logs collection: {config['logs_collection']} | State: {config['state_collection']}")
    print(f"Classify API: {config['classify_api_url']}")
    if config["max_jobs"]:
        print(f"MAX_JOBS={config['max_jobs']}")

    process_classification(config)


if __name__ == "__main__":
    main()

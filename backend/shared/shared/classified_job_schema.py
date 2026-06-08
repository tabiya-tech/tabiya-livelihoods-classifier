"""Scraper → classified_jobs field passthrough (align with scraper JOB_SCHEMA enrichment)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Mirror scraper/post_processing/integrate.py ENRICHMENT_FIELD_NAMES
SCRAPER_ENRICHMENT_FIELD_NAMES: tuple[str, ...] = (
    "city",
    "province",
    "sector",
    "domain",
    "isco_major_group",
    "isco_major_group_label",
    "isco_sub_major_group",
    "isco_sub_major_group_label",
    "is_priority_sector",
    "zqf_min",
    "zqf_max",
    "zqf_min_label",
    "zqf_max_label",
)


def copy_enrichment_fields(job: dict[str, Any]) -> dict[str, Any]:
    """Copy post-processing enrichment keys from a scraped job when present."""
    out: dict[str, Any] = {}
    for key in SCRAPER_ENRICHMENT_FIELD_NAMES:
        val = job.get(key)
        if val is not None:
            out[key] = val
    return out


def build_classified_job_document(
    job: dict[str, Any],
    *,
    job_id: Any,
    classification: dict[str, Any],
    metadata: dict[str, Any],
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Build the Mongo document written to classified_jobs."""
    now_utc = now_utc or datetime.now(timezone.utc)
    doc = {
        "job_id": job_id,
        "job_fingerprint": job.get("job_fingerprint"),
        "title": job.get("title"),
        "employer": job.get("employer"),
        "location": job.get("location"),
        "employment_type": job.get("employment_type"),
        "category": job.get("category"),
        "source_platform": job.get("source_platform"),
        "posted_date": job.get("posted_date"),
        "closing_date": job.get("closing_date") or job.get("expiry_date"),
        "application_url": job.get("application_url"),
        "description": job.get("description"),
        "requirements": job.get("requirements"),
        "salary_text": job.get("salary_text") or job.get("salary"),
        "classification": classification,
        "metadata": metadata,
        "updated_at": now_utc,
        **copy_enrichment_fields(job),
    }
    tr_src = job.get("translation")
    if isinstance(tr_src, dict) and tr_src:
        doc["translation"] = tr_src
    return doc

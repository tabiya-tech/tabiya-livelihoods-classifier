"""Tests for scraper enrichment passthrough into classified_jobs."""

from __future__ import annotations

from datetime import datetime, timezone

from backend.shared.shared.classified_job_schema import (
    SCRAPER_ENRICHMENT_FIELD_NAMES,
    build_classified_job_document,
    copy_enrichment_fields,
)


def test_copy_enrichment_fields_copies_present_values_only() -> None:
    job = {
        "city": "Lusaka",
        "province": "Lusaka Province",
        "sector": "Health",
        "zqf_min": 4,
        "title": "Nurse",
    }
    out = copy_enrichment_fields(job)
    assert out == {
        "city": "Lusaka",
        "province": "Lusaka Province",
        "sector": "Health",
        "zqf_min": 4,
    }
    for key in SCRAPER_ENRICHMENT_FIELD_NAMES:
        if key not in out:
            assert job.get(key) is None


def test_build_classified_job_document_includes_enrichment_fields() -> None:
    now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
    job = {
        "_id": "abc",
        "job_fingerprint": "fp1",
        "title": "Software Engineer",
        "employer": "Acme",
        "location": "Nairobi",
        "city": ["Nairobi"],
        "province": "Nairobi County",
        "sector": "ICT",
        "domain": "Software Development",
        "isco_major_group": 2,
        "isco_major_group_label": "Professionals",
        "is_priority_sector": True,
    }
    doc = build_classified_job_document(
        job,
        job_id=job["_id"],
        classification={"entities": [], "entity_counts": {}},
        metadata={"classifier_version": "1.0.0"},
        now_utc=now,
    )
    assert doc["job_id"] == "abc"
    assert doc["city"] == ["Nairobi"]
    assert doc["province"] == "Nairobi County"
    assert doc["sector"] == "ICT"
    assert doc["domain"] == "Software Development"
    assert doc["isco_major_group"] == 2
    assert doc["isco_major_group_label"] == "Professionals"
    assert doc["is_priority_sector"] is True
    assert doc["updated_at"] == now

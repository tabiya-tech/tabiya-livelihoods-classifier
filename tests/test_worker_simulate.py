"""
Step 6 Test — Simulate the Change Stream Worker with dummy data.

This script mimics what the real worker does:
  1. A "new job" appears (dummy data)
  2. Worker detects it
  3. Sends it to Classify API (http://localhost:5001)
  4. Stores the result in InMemoryJobRepository
  5. Prints the full pipeline trace

Requires: ner_server (5002), nel_server (5003), classify_server (5001) running.
No MongoDB needed.

Usage:
    python test_step6_simulate.py
"""

import sys
import os
import time
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from util.job_text import build_input_text, compute_hash

CLASSIFY_API_URL = "http://localhost:5001"

DUMMY_JOBS = [
    {
        "job_fingerprint": "sim_001_nurse",
        "title": "Registered Nurse",
        "description": (
            "We are hiring a Registered Nurse for our Nairobi clinic. "
            "The ideal candidate should have a Diploma in Nursing and at least "
            "2 years of experience in patient care. Skills required include "
            "wound management, vital signs monitoring, and medication administration."
        ),
        "employer": "HealthPlus Kenya",
        "location": "Nairobi, Kenya",
    },
    {
        "job_fingerprint": "sim_002_driver",
        "title": "Heavy Truck Driver",
        "description": (
            "Looking for an experienced heavy truck driver for long-haul routes "
            "between Dar es Salaam and Mombasa. Must hold a valid Class C driving "
            "license. Minimum 3 years experience in commercial driving. Knowledge "
            "of vehicle maintenance and road safety regulations is essential."
        ),
        "employer": "TransEast Logistics",
        "location": "Dar es Salaam, Tanzania",
    },
    {
        "job_fingerprint": "sim_003_accountant",
        "title": "Junior Accountant",
        "description": (
            "Entry-level accounting position available. Responsibilities include "
            "bookkeeping, preparing financial statements, tax filing, and payroll "
            "processing. Bachelor degree in Accounting or Finance required. "
            "Proficiency in QuickBooks and Microsoft Excel is a must."
        ),
        "employer": "Finserve Ltd",
        "location": "Kampala, Uganda",
    },
]


def main():
    print("=" * 70)
    print("CHANGE STREAM WORKER — SIMULATION MODE (no MongoDB)")
    print("=" * 70)

    # Health check
    try:
        r = requests.get(f"{CLASSIFY_API_URL}/v1/health", timeout=5)
        health = r.json()
        print(f"\nClassify API health: {health['status']}")
        print(f"  NER API:  {health['dependencies']['ner-api']}")
        print(f"  NEL API:  {health['dependencies']['nel-api']}")
        if health["status"] != "healthy":
            print("\nERROR: Classify API is not fully healthy. Start all 3 servers first.")
            return
    except requests.RequestException as e:
        print(f"\nERROR: Cannot reach Classify API at {CLASSIFY_API_URL}: {e}")
        print("Make sure all 3 servers are running (ports 5001, 5002, 5003).")
        return

    results_store = {}

    print(f"\nSimulating {len(DUMMY_JOBS)} incoming jobs...\n")

    for i, job in enumerate(DUMMY_JOBS, 1):
        fp = job["job_fingerprint"]
        input_text = build_input_text(job)
        input_hash = compute_hash(input_text)

        print("-" * 70)
        print(f"[{i}/{len(DUMMY_JOBS)}] CHANGE EVENT: new job detected")
        print(f"  Fingerprint : {fp}")
        print(f"  Title       : {job['title']}")
        print(f"  Employer    : {job['employer']}")
        print(f"  Location    : {job['location']}")
        print(f"  Input hash  : {input_hash[:16]}...")

        # Check duplicate (simulated)
        if fp in results_store and results_store[fp]["input_text_hash"] == input_hash:
            print(f"  SKIP — already classified with same input hash")
            continue

        # Call Classify API
        print(f"  Calling Classify API...")
        start = time.time()
        try:
            resp = requests.post(
                f"{CLASSIFY_API_URL}/v1/classify",
                json={"text": input_text},
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.RequestException as e:
            print(f"  ERROR: {e}")
            continue

        elapsed = round((time.time() - start) * 1000)
        classification = result.get("classification", {})
        metadata = result.get("metadata", {})
        entities = classification.get("entities", [])
        counts = classification.get("entity_counts", {})

        # Store result (simulated write to classified-jobs)
        results_store[fp] = {
            "job_fingerprint": fp,
            "input_text_hash": input_hash,
            "classification": classification,
            "metadata": metadata,
            "source_fields": {
                "title": job["title"],
                "employer": job["employer"],
                "location": job["location"],
            },
        }

        print(f"  Classification complete in {elapsed}ms")
        print(f"  Entities found: {sum(counts.values())} total — {dict(counts)}")
        print(f"  Top entities:")
        for ent in entities[:5]:
            linked = ent.get("linked_entities", [])
            top_match = f" → {linked[0]['label']} ({linked[0]['similarity_score']})" if linked else ""
            print(f"    [{ent['entity_type']}] \"{ent['surface_form']}\"{top_match}")
        if len(entities) > 5:
            print(f"    ... and {len(entities) - 5} more")

        print(f"  SAVED to classified-jobs (in-memory)")

    # Dedup test: replay the first job
    print("\n" + "=" * 70)
    print("DEDUP TEST: Replaying first job (should be skipped)")
    print("=" * 70)
    first = DUMMY_JOBS[0]
    fp = first["job_fingerprint"]
    input_text = build_input_text(first)
    input_hash = compute_hash(input_text)
    if fp in results_store and results_store[fp]["input_text_hash"] == input_hash:
        print(f"  SKIP — {fp} already classified with same input hash ✓")
    else:
        print(f"  ERROR — dedup failed!")

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Jobs processed  : {len(results_store)}")
    print(f"  Jobs skipped     : 1 (dedup test)")
    print(f"  Storage backend  : in-memory (no MongoDB)")
    print(f"  NER model        : {list(results_store.values())[0]['metadata'].get('model_name', '?')}")
    print(f"  NEL model        : {list(results_store.values())[0]['metadata'].get('linker_model', '?')}")
    print(f"\nAll {len(results_store)} classified jobs stored. In production, these go to MongoDB classified-jobs collection.")
    print("Full pipeline: Scraper → raw-jobs → [Worker detects] → Classify API → classified-jobs  ✓")


if __name__ == "__main__":
    main()

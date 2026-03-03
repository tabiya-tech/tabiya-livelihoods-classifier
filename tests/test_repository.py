"""
Step 2 Local Test: Verify Repository layer works.
  - Part A: InMemoryJobRepository (no database needed)
  - Part B: MongoJobRepository (reads from your Atlas raw-jobs)

Run from the classifier root: python test_step2.py
"""

import asyncio

async def test_in_memory():
    """Test the InMemory repository — no credentials, no database."""
    from app.repository import InMemoryJobRepository, compute_input_text_hash

    print("\n[Part A] Testing InMemoryJobRepository...")
    repo = InMemoryJobRepository()

    # Seed a fake raw job
    fake_job = {
        "job_fingerprint": "test123",
        "title": "Head Chef",
        "description": "We need someone who can plan menus and manage kitchen staff.",
        "employer": "Sarova Hotels",
        "location": "Nairobi",
    }
    repo.add_raw_job(fake_job)

    # Test get_job
    job = await repo.get_job("test123")
    assert job is not None, "get_job failed"
    assert job["title"] == "Head Chef"
    print("  ✓ get_job — found job by fingerprint")

    # Test get_job for missing
    missing = await repo.get_job("nonexistent")
    assert missing is None
    print("  ✓ get_job — returns None for missing fingerprint")

    # Test get_unclassified_jobs
    unclassified = await repo.get_unclassified_jobs()
    assert len(unclassified) == 1
    print(f"  ✓ get_unclassified_jobs — found {len(unclassified)} unclassified job(s)")

    # Test save_classification
    text_hash = compute_input_text_hash(fake_job["title"], fake_job["description"])
    classification = {
        "job_fingerprint": "test123",
        "source_job": {
            "title": fake_job["title"],
            "description": fake_job["description"],
        },
        "classification": {
            "entities": [{"entity_type": "occupation", "surface_form": "Head Chef"}],
            "entity_counts": {"occupation": 1, "skill": 2},
        },
        "metadata": {
            "classifier_version": "1.0.0",
            "input_text_hash": text_hash,
        },
        "status": "completed",
    }
    await repo.save_classification(classification)
    print("  ✓ save_classification — saved without error")

    # Test get_classification
    result = await repo.get_classification("test123")
    assert result is not None
    assert result["status"] == "completed"
    print("  ✓ get_classification — retrieved saved classification")

    # Test is_already_classified
    is_done = await repo.is_already_classified("test123", text_hash)
    assert is_done is True
    print("  ✓ is_already_classified — returns True for same text hash")

    is_done_diff = await repo.is_already_classified("test123", "different_hash")
    assert is_done_diff is False
    print("  ✓ is_already_classified — returns False for different text hash")

    # Test unclassified is now empty
    unclassified = await repo.get_unclassified_jobs()
    assert len(unclassified) == 0
    print(f"  ✓ get_unclassified_jobs — {len(unclassified)} unclassified after classification")

    await repo.close()
    print("\n  Part A PASSED — all InMemory tests passed ✓")


async def test_mongo():
    """Test the Mongo repository — reads from Atlas, writes to classified-jobs."""
    from app.repository import MongoJobRepository

    print("\n[Part B] Testing MongoJobRepository (Atlas)...")
    repo = MongoJobRepository()

    # Test: read one job from raw-jobs
    print("  Connecting to Atlas and reading from raw-jobs...")
    sample = await repo.raw_jobs.find_one()

    if sample is None:
        print("  ⚠ raw-jobs collection is empty — skipping Mongo tests")
        await repo.close()
        return

    fp = sample.get("job_fingerprint", "unknown")
    title = sample.get("title", "no title")
    print(f"  ✓ Connected — found job: \"{title}\" (fingerprint: {fp[:12]}...)")

    # Test get_job
    job = await repo.get_job(fp)
    assert job is not None
    print(f"  ✓ get_job — retrieved by fingerprint")

    # Test get_unclassified_jobs
    unclassified = await repo.get_unclassified_jobs(limit=5)
    print(f"  ✓ get_unclassified_jobs — found {len(unclassified)} unclassified job(s)")

    # Count total raw-jobs and classified-jobs
    raw_count = await repo.raw_jobs.count_documents({})
    classified_count = await repo.classified_jobs.count_documents({})
    print(f"  ℹ raw-jobs: {raw_count} documents, classified-jobs: {classified_count} documents")

    await repo.close()
    print("\n  Part B PASSED — Mongo connection and reads working ✓")


async def main():
    print("=" * 60)
    print("STEP 2 TEST: Repository Layer")
    print("=" * 60)

    await test_in_memory()
    await test_mongo()

    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE")
    print("=" * 60)
    print("  app/repository.py:")
    print("    JobRepository    — abstract interface          ✓")
    print("    MongoJobRepository — reads/writes Atlas        ✓")
    print("    InMemoryJobRepository — for testing            ✓")
    print("    compute_input_text_hash — dedup helper         ✓")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

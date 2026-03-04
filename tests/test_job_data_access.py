"""
Local test: Verify Repository layer works.
  - Part A: InMemoryJobRepository (no database needed)
  - Part B: MongoJobRepository (reads from your Atlas raw-jobs)

All synchronous — matches the Flask-based codebase.
Run from the classifier root: python -m tests.test_repository
"""


def test_in_memory():
    """Test the InMemory repository — no credentials, no database."""
    from app.job_data_access import InMemoryJobRepository, compute_input_text_hash

    print("\n[Part A] Testing InMemoryJobRepository...")
    repo = InMemoryJobRepository()

    fake_job = {
        "job_fingerprint": "test123",
        "title": "Head Chef",
        "description": "We need someone who can plan menus and manage kitchen staff.",
        "employer": "Sarova Hotels",
        "location": "Nairobi",
    }
    repo.add_raw_job(fake_job)

    job = repo.get_job("test123")
    assert job is not None, "get_job failed"
    assert job["title"] == "Head Chef"
    print("  ok get_job — found job by fingerprint")

    missing = repo.get_job("nonexistent")
    assert missing is None
    print("  ok get_job — returns None for missing fingerprint")

    unclassified = repo.get_unclassified_jobs()
    assert len(unclassified) == 1
    print(f"  ok get_unclassified_jobs — found {len(unclassified)} unclassified job(s)")

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
    repo.save_classification(classification)
    print("  ok save_classification — saved without error")

    result = repo.get_classification("test123")
    assert result is not None
    assert result["status"] == "completed"
    print("  ok get_classification — retrieved saved classification")

    is_done = repo.is_already_classified("test123", text_hash)
    assert is_done is True
    print("  ok is_already_classified — returns True for same text hash")

    is_done_diff = repo.is_already_classified("test123", "different_hash")
    assert is_done_diff is False
    print("  ok is_already_classified — returns False for different text hash")

    unclassified = repo.get_unclassified_jobs()
    assert len(unclassified) == 0
    print(f"  ok get_unclassified_jobs — {len(unclassified)} unclassified after classification")

    repo.close()
    print("\n  Part A PASSED — all InMemory tests passed")


def test_mongo():
    """Test the Mongo repository — reads from Atlas."""
    from app.job_data_access import MongoJobRepository

    print("\n[Part B] Testing MongoJobRepository (Atlas)...")
    repo = MongoJobRepository()

    print("  Connecting to Atlas and reading from raw-jobs...")
    sample = repo.raw_jobs.find_one()

    if sample is None:
        print("  WARNING: raw-jobs collection is empty — skipping Mongo tests")
        repo.close()
        return

    fp = sample.get("job_fingerprint", "unknown")
    title = sample.get("title", "no title")
    print(f"  ok Connected — found job: \"{title}\" (fingerprint: {fp[:12]}...)")

    job = repo.get_job(fp)
    assert job is not None
    print("  ok get_job — retrieved by fingerprint")

    unclassified = repo.get_unclassified_jobs(limit=5)
    print(f"  ok get_unclassified_jobs — found {len(unclassified)} unclassified job(s)")

    raw_count = repo.raw_jobs.count_documents({})
    classified_count = repo.classified_jobs.count_documents({})
    print(f"  info raw-jobs: {raw_count} documents, classified-jobs: {classified_count} documents")

    repo.close()
    print("\n  Part B PASSED — Mongo connection and reads working")


def main():
    print("=" * 60)
    print("TEST: Repository Layer")
    print("=" * 60)

    test_in_memory()
    test_mongo()

    print("\n" + "=" * 60)
    print("REPOSITORY TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

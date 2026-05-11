"""Client-facing error messages: hide internal exceptions unless running locally."""

import os

_CLASSIFY_UPSTREAM_GENERIC = (
    "Classification temporarily unavailable. Please try again later."
)
_BATCH_JOB_GENERIC = "Classification failed for this job. Please try again later."
_BATCH_FATAL_GENERIC = "Batch processing failed."


def _expose_internal_errors() -> bool:
    return os.getenv("TARGET_ENVIRONMENT_TYPE", "").strip().lower() == "local"


def classify_upstream_public_detail(exc: BaseException) -> str:
    """Message for HTTP 502 on /v1/classify when NER/NEL/orchestration fails."""
    if _expose_internal_errors():
        return str(exc)
    return _CLASSIFY_UPSTREAM_GENERIC


def batch_job_public_error(exc: BaseException) -> str:
    """Per-job `error` field in batch results when classify raises."""
    if _expose_internal_errors():
        return str(exc)
    return _BATCH_JOB_GENERIC


def batch_fatal_public_message(exc: BaseException) -> str:
    """Stored on the batch document when the batch runner crashes."""
    if _expose_internal_errors():
        return str(exc)
    return _BATCH_FATAL_GENERIC

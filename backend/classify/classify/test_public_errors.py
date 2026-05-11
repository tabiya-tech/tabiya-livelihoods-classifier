"""Tests for client-facing error sanitization (point B)."""

import pytest

from classify.public_errors import (
    batch_fatal_public_message,
    batch_job_public_error,
    classify_upstream_public_detail,
)


class TestPublicErrors:
    def test_classify_upstream_sanitized_when_not_local(self, monkeypatch):
        monkeypatch.delenv("TARGET_ENVIRONMENT_TYPE", raising=False)
        exc = Exception("NER connection reset by peer")
        detail = classify_upstream_public_detail(exc)
        assert "NER" not in detail
        assert "temporarily unavailable" in detail.lower()

    def test_classify_upstream_exposes_when_local(self, monkeypatch):
        monkeypatch.setenv("TARGET_ENVIRONMENT_TYPE", "local")
        exc = Exception("NER connection reset by peer")
        assert classify_upstream_public_detail(exc) == "NER connection reset by peer"

    def test_batch_job_error_sanitized_when_not_local(self, monkeypatch):
        monkeypatch.delenv("TARGET_ENVIRONMENT_TYPE", raising=False)
        exc = RuntimeError("internal stack trace here")
        msg = batch_job_public_error(exc)
        assert "stack" not in msg.lower()
        assert "try again" in msg.lower()

    def test_batch_job_error_exposes_when_local(self, monkeypatch):
        monkeypatch.setenv("TARGET_ENVIRONMENT_TYPE", "local")
        exc = RuntimeError("internal stack trace here")
        assert batch_job_public_error(exc) == "internal stack trace here"

    def test_batch_fatal_sanitized_when_not_local(self, monkeypatch):
        monkeypatch.delenv("TARGET_ENVIRONMENT_TYPE", raising=False)
        exc = ValueError("Mongo cursor died")
        msg = batch_fatal_public_message(exc)
        assert "Mongo" not in msg
        assert "failed" in msg.lower()

    def test_batch_fatal_exposes_when_local(self, monkeypatch):
        monkeypatch.setenv("TARGET_ENVIRONMENT_TYPE", "local")
        exc = ValueError("Mongo cursor died")
        assert batch_fatal_public_message(exc) == "Mongo cursor died"

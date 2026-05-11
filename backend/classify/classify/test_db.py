"""Tests for MongoDB client wiring (point G — TLS defaults)."""

from unittest.mock import patch

import classify.config as classify_config
from classify.db import _create_db


class TestCreateDbTls:
    def test_default_does_not_pass_tls_allow_invalid(self, monkeypatch):
        monkeypatch.setattr(classify_config, "MONGODB_TLS_ALLOW_INVALID", False)
        with patch("classify.db.AsyncIOMotorClient") as mock_client:
            _create_db("mongodb://localhost:27017", "test")
        mock_client.assert_called_once_with("mongodb://localhost:27017")
        assert "tlsAllowInvalidCertificates" not in mock_client.call_args.kwargs

    def test_opt_in_passes_tls_allow_invalid(self, monkeypatch):
        monkeypatch.setattr(classify_config, "MONGODB_TLS_ALLOW_INVALID", True)
        with patch("classify.db.AsyncIOMotorClient") as mock_client:
            _create_db("mongodb://localhost:27017", "test")
        mock_client.assert_called_once_with(
            "mongodb://localhost:27017",
            tlsAllowInvalidCertificates=True,
        )

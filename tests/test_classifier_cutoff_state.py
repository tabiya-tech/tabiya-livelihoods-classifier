"""Classifier incremental cutoff: ``last_cutoff_date`` used exactly (no −1 day)."""

from __future__ import annotations

from datetime import datetime, timezone

import run_classifier as rc


def test_cutoff_str_from_stored_last_string_exact() -> None:
    assert rc._cutoff_str_from_stored_last("2026-04-15") == "2026-04-15"


def test_cutoff_str_from_stored_last_datetime_utc() -> None:
    dt = datetime(2026, 3, 1, 15, 30, tzinfo=timezone.utc)
    assert rc._cutoff_str_from_stored_last(dt) == "2026-03-01"


def test_cutoff_str_no_minus_one_day_regression() -> None:
    """Regression guard: must not return 2026-04-14 for stored 2026-04-15."""
    assert rc._cutoff_str_from_stored_last("2026-04-15") != "2026-04-14"


def test_cutoff_str_invalid_returns_none() -> None:
    assert rc._cutoff_str_from_stored_last(None) is None
    assert rc._cutoff_str_from_stored_last("not-a-date") is None

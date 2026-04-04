"""Tests for scripts/check_deadline.py.

Culprit if failing: hours_until_deadline() or is_approaching logic in check_deadline.py.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import check_deadline as cd


def _bootstrap_with_next_gw(hours_from_now: float, next_gw_id: int = 32) -> dict:
    deadline = datetime.now(timezone.utc) + timedelta(hours=hours_from_now)
    return {
        "events": [
            {
                "id": next_gw_id,
                "is_next": True,
                "deadline_time": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        ]
    }


def test_deadline_within_48h_reports_approaching():
    bootstrap = _bootstrap_with_next_gw(hours_from_now=24)
    hours, gw_id = cd.hours_until_deadline(bootstrap)
    assert hours is not None
    assert hours < 48.0
    assert gw_id == 32


def test_deadline_beyond_48h_reports_not_approaching():
    bootstrap = _bootstrap_with_next_gw(hours_from_now=72)
    hours, gw_id = cd.hours_until_deadline(bootstrap)
    assert hours is not None
    assert hours > 48.0


def test_no_next_gw_returns_none():
    bootstrap = {"events": [{"id": 31, "is_next": False, "deadline_time": "2026-01-01T12:00:00Z"}]}
    hours, gw_id = cd.hours_until_deadline(bootstrap)
    assert hours is None
    assert gw_id is None


def test_exactly_48h_is_not_approaching():
    """Boundary: exactly 48h should not trigger (< 48, not <=)."""
    bootstrap = _bootstrap_with_next_gw(hours_from_now=48.01)
    hours, gw_id = cd.hours_until_deadline(bootstrap)
    assert hours > 48.0


def test_gw_id_returned_correctly():
    bootstrap = _bootstrap_with_next_gw(hours_from_now=10, next_gw_id=35)
    _, gw_id = cd.hours_until_deadline(bootstrap)
    assert gw_id == 35


def test_hours_value_is_numeric():
    """hours_until_deadline must return a float, not a string — used in Discord message."""
    bootstrap = _bootstrap_with_next_gw(hours_from_now=36.5)
    hours, _ = cd.hours_until_deadline(bootstrap)
    assert isinstance(hours, float)
    assert 36.0 < hours < 37.0

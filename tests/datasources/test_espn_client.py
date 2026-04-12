"""Tests for espn_client.py — all HTTP calls are mocked.

Contract verified:
  H-C5-T1  resolve_espn_player_id: direct map lookup returns correct ESPN ID
  H-C5-T2  resolve_espn_player_id: fuzzy match above threshold returns ESPN ID
  H-C5-T3  resolve_espn_player_id: unresolvable player returns None and logs to unresolved CSV
  H-C5-T4  fetch_espn_player_season: filters out eng.1 (PL) events
  H-C5-T5  fetch_espn_player_season: returns correct output columns
  H-C5-T6  fetch_espn_player_season: returns cached DataFrame on second call (no HTTP)
  H-C5-T7  fetch_espn_recent: returns only matches within the look-back window
  H-C5-T8  _name_similarity: basic correctness
"""
from __future__ import annotations

import csv
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import src.pipeline.datasources.espn_client as ec


# ── Fixtures ───────────────────────────────────────────────────────────────────

SEED_ROWS = [
    {
        "fpl_code": "244851",
        "web_name": "Palmer",
        "espn_id": "296395",
        "espn_name": "Cole Palmer",
        "verified": "true",
    },
    {
        "fpl_code": "60799",
        "web_name": "Enzo Fernández",
        "espn_id": "285450",
        "espn_name": "Enzo Fernandez",
        "verified": "true",
    },
]


def _write_seed_map(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fpl_code", "web_name", "espn_id", "espn_name", "verified"])
        writer.writeheader()
        writer.writerows(SEED_ROWS)


def _mock_eventlog_response() -> dict:
    """Minimal ESPN eventlog response with one UCL and one PL event."""
    return {
        "events": {
            "items": [
                {
                    "$ref": "http://sports.core.api.espn.com/v2/sports/soccer/uefa.champions/events/697123/competitors/296395/statistics/0",
                    "date": "2024-10-02T19:00:00Z",
                },
                {
                    "$ref": "http://sports.core.api.espn.com/v2/sports/soccer/eng.1/events/900001/competitors/296395/statistics/0",
                    "date": "2024-10-06T14:00:00Z",
                },
            ]
        }
    }


def _mock_summary_response(espn_id: int) -> dict:
    """Minimal ESPN event summary with one player stats block."""
    return {
        "boxscore": {
            "players": [
                {
                    "statistics": [
                        {
                            "athletes": [
                                {
                                    "athlete": {"id": str(espn_id)},
                                    "names": ["minutesPlayed", "goals", "assists"],
                                    "stats": ["72", "1", "0"],
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestResolveEspnPlayerId:
    def test_direct_lookup_returns_correct_id(self, tmp_path):
        """H-C5-T1: fpl_code present in map → return ESPN ID immediately."""
        id_map_path = tmp_path / "espn_player_id_map.csv"
        _write_seed_map(id_map_path)

        with patch.object(ec, "_ID_MAP_PATH", id_map_path):
            result = ec.resolve_espn_player_id(244851, "Palmer", "Palmer")

        assert result == 296395

    def test_fuzzy_match_above_threshold_returns_id(self, tmp_path):
        """H-C5-T2: unrecognised fpl_code but matching name → fuzzy resolves it."""
        id_map_path = tmp_path / "espn_player_id_map.csv"
        _write_seed_map(id_map_path)

        unresolved = tmp_path / "espn_unresolved.csv"
        with patch.object(ec, "_ID_MAP_PATH", id_map_path):
            with patch.object(ec, "_UNRESOLVED_PATH", unresolved):
                # fpl_code 99999 not in map; name "Cole Palmer" should fuzzy-match
                result = ec.resolve_espn_player_id(99999, "Cole Palmer", "Palmer")

        assert result == 296395

    def test_unresolvable_returns_none_and_logs(self, tmp_path):
        """H-C5-T3: no match → return None and write to unresolved CSV."""
        id_map_path = tmp_path / "espn_player_id_map.csv"
        _write_seed_map(id_map_path)
        unresolved = tmp_path / "espn_unresolved.csv"

        with patch.object(ec, "_ID_MAP_PATH", id_map_path):
            with patch.object(ec, "_UNRESOLVED_PATH", unresolved):
                result = ec.resolve_espn_player_id(11111, "Completely Unique Player", "Nobody")

        assert result is None
        assert unresolved.exists()
        rows = list(csv.DictReader(unresolved.open()))
        assert any(r["fpl_code"] == "11111" for r in rows)


class TestFetchEspnPlayerSeason:
    def test_filters_pl_events(self, tmp_path):
        """H-C5-T4: eng.1 events must be excluded from output."""
        cache_dir = tmp_path / "espn_cache"
        unresolved = tmp_path / "espn_unresolved.csv"

        event_log = _mock_eventlog_response()
        summary = _mock_summary_response(296395)

        with patch.object(ec, "_CACHE_DIR", cache_dir):
            with patch.object(ec, "_UNRESOLVED_PATH", unresolved):
                with patch.object(ec, "_get_json", side_effect=[event_log, summary]):
                    with patch("time.sleep"):
                        df = ec.fetch_espn_player_season(296395, 2024, fpl_code=244851)

        # Only the UCL event should remain
        assert len(df) == 1
        assert df.iloc[0]["league_slug"] == "uefa.champions"

    def test_returns_correct_columns(self, tmp_path):
        """H-C5-T5: output DataFrame must have exactly the expected columns."""
        cache_dir = tmp_path / "espn_cache"
        event_log = _mock_eventlog_response()
        summary = _mock_summary_response(296395)

        with patch.object(ec, "_CACHE_DIR", cache_dir):
            with patch.object(ec, "_get_json", side_effect=[event_log, summary]):
                with patch("time.sleep"):
                    df = ec.fetch_espn_player_season(296395, 2024)

        for col in ec._OUTPUT_COLS:
            assert col in df.columns, f"Missing expected column: {col}"

    def test_caches_on_first_call_and_reuses(self, tmp_path):
        """H-C5-T6: second call reads from cache — _get_json not called again."""
        cache_dir = tmp_path / "espn_cache"
        event_log = _mock_eventlog_response()
        summary = _mock_summary_response(296395)

        with patch.object(ec, "_CACHE_DIR", cache_dir):
            with patch.object(ec, "_get_json", side_effect=[event_log, summary]) as mock_get:
                with patch("time.sleep"):
                    df1 = ec.fetch_espn_player_season(296395, 2024)

            call_count_after_first = mock_get.call_count

            # Second call — should hit cache
            with patch.object(ec, "_get_json") as mock_get2:
                df2 = ec.fetch_espn_player_season(296395, 2024)
                assert mock_get2.call_count == 0, "Should have used cache, not HTTP"

        assert len(df1) == len(df2)


class TestFetchEspnRecent:
    def test_filters_by_date(self, tmp_path):
        """H-C5-T7: only matches within look-back window are returned."""
        cache_dir = tmp_path / "espn_cache"

        # Create a fake cached season file with one old and one recent match
        from datetime import date, timedelta
        today = date.today()
        recent_date = (today - timedelta(days=10)).isoformat()
        old_date = (today - timedelta(days=60)).isoformat()

        season_year = today.year if today.month >= 8 else today.year - 1
        cache_file = tmp_path / "espn_cache" / f"player_296395_season_{season_year}.csv"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        rows = [
            {col: ("296395" if col == "espn_id" else (recent_date if col == "match_date" else "0"))
             for col in ec._OUTPUT_COLS},
            {col: ("296395" if col == "espn_id" else (old_date if col == "match_date" else "0"))
             for col in ec._OUTPUT_COLS},
        ]
        pd.DataFrame(rows).to_csv(cache_file, index=False)

        with patch.object(ec, "_CACHE_DIR", cache_dir):
            df = ec.fetch_espn_recent(296395, days=30)

        assert len(df) == 1
        assert recent_date in str(df.iloc[0]["match_date"])


class TestNameSimilarity:
    def test_identical_strings(self):
        """H-C5-T8: identical strings → score 1.0."""
        assert ec._name_similarity("cole palmer", "cole palmer") == 1.0

    def test_dissimilar_strings(self):
        assert ec._name_similarity("cole palmer", "erling haaland") < 0.3

    def test_close_strings(self):
        # "cole palmer" vs "cole palme" — should be high
        assert ec._name_similarity("cole palmer", "cole palme") > 0.7

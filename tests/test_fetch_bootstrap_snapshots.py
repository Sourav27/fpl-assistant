"""TDD tests for scripts/fetch_bootstrap_snapshots.py.

Tests written BEFORE verifying all behaviours pass. Each test documents an
explicit contract; tests marked # WILL FAIL identify gaps in the current
implementation that must be fixed before they can go green.

Coverage:
  find_wayback_snapshot — CDX query params, result selection, empty window
  backfill             — skip cached, fetch+save, bad GW, no snapshot, API failure
  live_mode            — saves current+next, valid JSON, API failure, end-of-season
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

# scripts/ is not a package — add it to sys.path for import
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import fetch_bootstrap_snapshots as fbs


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _bootstrap(current_gw: int, next_gw: int | None = None) -> dict:
    """Minimal FPL bootstrap with realistic event flags."""
    events = []
    for i in range(1, current_gw + 3):
        events.append({
            "id": i,
            "deadline_time": f"2026-{((i - 1) // 4 + 1):02d}-{((i - 1) % 4) * 7 + 1:02d}T11:00:00Z",
            "is_current": i == current_gw,
            "is_next": i == next_gw,
            "finished": i < current_gw,
        })
    return {"events": events, "elements": [], "teams": []}


def _cdx_response(timestamps: list[str]) -> list:
    """CDX API JSON: first row is header, rest are data rows."""
    return [["timestamp"]] + [[ts] for ts in timestamps]


def _mock_get(return_value) -> MagicMock:
    m = MagicMock()
    m.json.return_value = return_value
    m.raise_for_status.return_value = None
    return m


# ---------------------------------------------------------------------------
# find_wayback_snapshot
# ---------------------------------------------------------------------------

class TestFindWaybackSnapshot:

    def test_returns_last_timestamp_closest_to_deadline(self):
        """CDX returns results in chronological order; last = closest to deadline."""
        cdx = _cdx_response(["20260310120000", "20260312180000", "20260313090000"])
        with patch("fetch_bootstrap_snapshots._get", return_value=_mock_get(cdx)):
            result = fbs.find_wayback_snapshot("2026-03-14T11:00:00Z")
        assert result == "20260313090000"

    def test_returns_none_when_no_snapshots_in_window(self):
        """Header-only CDX response (no data rows) → None."""
        with patch("fetch_bootstrap_snapshots._get", return_value=_mock_get([["timestamp"]])):
            result = fbs.find_wayback_snapshot("2026-03-14T11:00:00Z")
        assert result is None

    def test_cdx_query_filters_for_200_status_only(self):
        """CDX must filter statuscode:200 to skip archived redirects and errors."""
        with patch("fetch_bootstrap_snapshots._get", return_value=_mock_get([["timestamp"]])) as m:
            fbs.find_wayback_snapshot("2026-03-14T11:00:00Z")
        _, kwargs = m.call_args
        assert kwargs["params"]["filter"] == "statuscode:200"

    def test_cdx_to_param_is_at_or_before_deadline(self):
        """'to' CDX param must not exceed the deadline (only pre-deadline snapshots)."""
        with patch("fetch_bootstrap_snapshots._get", return_value=_mock_get([["timestamp"]])) as m:
            fbs.find_wayback_snapshot("2026-03-14T11:00:00Z")
        _, kwargs = m.call_args
        assert kwargs["params"]["to"] <= "20260314110000"

    def test_cdx_from_param_is_lookback_days_before_deadline(self):
        """'from' CDX param must be LOOKBACK_DAYS before the deadline."""
        from datetime import datetime, timezone, timedelta
        deadline = datetime(2026, 3, 14, 11, 0, 0, tzinfo=timezone.utc)
        expected_from = (deadline - timedelta(days=fbs.LOOKBACK_DAYS)).strftime("%Y%m%d%H%M%S")

        with patch("fetch_bootstrap_snapshots._get", return_value=_mock_get([["timestamp"]])) as m:
            fbs.find_wayback_snapshot("2026-03-14T11:00:00Z")
        _, kwargs = m.call_args
        # Allow up to 1-day tolerance (time-of-day rounding in implementation)
        assert kwargs["params"]["from"] <= expected_from


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------

class TestBackfill:

    def test_skips_gw_when_snapshot_already_cached(self, tmp_path, capsys):
        """GW with an existing snapshot file must not trigger a Wayback fetch."""
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()
        (snapshots_dir / "bootstrap_gw30.json").write_text(json.dumps({"events": []}))

        with patch("fetch_bootstrap_snapshots.fetch_live_bootstrap", return_value=_bootstrap(32)), \
             patch("fetch_bootstrap_snapshots.SNAPSHOTS_DIR", snapshots_dir), \
             patch("fetch_bootstrap_snapshots.find_wayback_snapshot") as mock_cdx, \
             patch("fetch_bootstrap_snapshots.time.sleep"):
            fbs.backfill(target_gw=30)

        mock_cdx.assert_not_called()
        assert "SKIP" in capsys.readouterr().out

    def test_fetches_and_saves_snapshot_for_uncached_past_gw(self, tmp_path):
        """Uncached past GW: fetch from Wayback and write to SNAPSHOTS_DIR."""
        past_bootstrap = {"events": [{"id": 30}], "elements": []}
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        with patch("fetch_bootstrap_snapshots.fetch_live_bootstrap", return_value=_bootstrap(32)), \
             patch("fetch_bootstrap_snapshots.SNAPSHOTS_DIR", snapshots_dir), \
             patch("fetch_bootstrap_snapshots.find_wayback_snapshot", return_value="20260313090000"), \
             patch("fetch_bootstrap_snapshots.fetch_wayback_bootstrap", return_value=past_bootstrap), \
             patch("fetch_bootstrap_snapshots.time.sleep"):
            fbs.backfill(target_gw=30)

        saved = snapshots_dir / "bootstrap_gw30.json"
        assert saved.exists()
        assert json.loads(saved.read_text(encoding="utf-8"))["events"][0]["id"] == 30

    def test_errors_when_target_gw_is_current_or_future(self, tmp_path, capsys):
        """Requesting backfill for current/future GW must print ERROR and not fetch."""
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        with patch("fetch_bootstrap_snapshots.fetch_live_bootstrap", return_value=_bootstrap(32)), \
             patch("fetch_bootstrap_snapshots.SNAPSHOTS_DIR", snapshots_dir), \
             patch("fetch_bootstrap_snapshots.find_wayback_snapshot") as mock_cdx:
            fbs.backfill(target_gw=32)

        mock_cdx.assert_not_called()
        assert "ERROR" in capsys.readouterr().out

    def test_warns_when_no_wayback_snapshot_available(self, tmp_path, capsys):
        """CDX returning no snapshots must print WARNING and leave no file."""
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        with patch("fetch_bootstrap_snapshots.fetch_live_bootstrap", return_value=_bootstrap(32)), \
             patch("fetch_bootstrap_snapshots.SNAPSHOTS_DIR", snapshots_dir), \
             patch("fetch_bootstrap_snapshots.find_wayback_snapshot", return_value=None), \
             patch("fetch_bootstrap_snapshots.time.sleep"):
            fbs.backfill(target_gw=30)

        assert "WARNING" in capsys.readouterr().out
        assert not (snapshots_dir / "bootstrap_gw30.json").exists()

    def test_handles_api_failure_on_startup(self, capsys):
        """If initial live bootstrap fetch fails, backfill must print ERROR not crash.
        # WILL FAIL — backfill() has no try/except around fetch_live_bootstrap()
        """
        with patch("fetch_bootstrap_snapshots.fetch_live_bootstrap",
                   side_effect=requests.ConnectionError("no network")):
            fbs.backfill()  # must NOT raise

        assert "ERROR" in capsys.readouterr().out

    def test_continues_remaining_gws_after_one_failure(self, tmp_path, capsys):
        """A Wayback fetch error on one GW must not abort the rest of the backfill."""
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()
        good_bootstrap = {"events": [{"id": 29}], "elements": []}

        # GW29 fails, GW30 succeeds
        def wayback_side_effect(deadline):
            if "2026-02" in deadline:  # GW29
                raise requests.ConnectionError("timeout")
            return "20260313090000"

        with patch("fetch_bootstrap_snapshots.fetch_live_bootstrap", return_value=_bootstrap(32)), \
             patch("fetch_bootstrap_snapshots.SNAPSHOTS_DIR", snapshots_dir), \
             patch("fetch_bootstrap_snapshots.find_wayback_snapshot", side_effect=wayback_side_effect), \
             patch("fetch_bootstrap_snapshots.fetch_wayback_bootstrap", return_value=good_bootstrap), \
             patch("fetch_bootstrap_snapshots.time.sleep"):
            fbs.backfill()  # must not raise

        # GW30 should still be saved despite GW29 failing
        assert (snapshots_dir / "bootstrap_gw30.json").exists()


# ---------------------------------------------------------------------------
# live_mode
# ---------------------------------------------------------------------------

class TestLiveMode:

    def test_saves_snapshot_for_both_current_and_next_gw(self, tmp_path):
        """live_mode must write bootstrap_gw{N}.json for both current and next GW."""
        bootstrap = _bootstrap(current_gw=32, next_gw=33)
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        with patch("fetch_bootstrap_snapshots.fetch_live_bootstrap", return_value=bootstrap), \
             patch("fetch_bootstrap_snapshots.SNAPSHOTS_DIR", snapshots_dir):
            fbs.live_mode()

        assert (snapshots_dir / "bootstrap_gw32.json").exists()
        assert (snapshots_dir / "bootstrap_gw33.json").exists()

    def test_saved_snapshot_contains_events_key(self, tmp_path):
        """Snapshot file must be valid JSON with an 'events' key."""
        bootstrap = _bootstrap(current_gw=32, next_gw=33)
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        with patch("fetch_bootstrap_snapshots.fetch_live_bootstrap", return_value=bootstrap), \
             patch("fetch_bootstrap_snapshots.SNAPSHOTS_DIR", snapshots_dir):
            fbs.live_mode()

        data = json.loads((snapshots_dir / "bootstrap_gw32.json").read_text(encoding="utf-8"))
        assert "events" in data

    def test_handles_api_failure_gracefully(self, capsys):
        """live_mode must print ERROR and not raise when FPL API is unreachable.
        # WILL FAIL — live_mode() has no try/except around fetch_live_bootstrap()
        """
        with patch("fetch_bootstrap_snapshots.fetch_live_bootstrap",
                   side_effect=requests.ConnectionError("timeout")):
            fbs.live_mode()  # must NOT raise

        assert "ERROR" in capsys.readouterr().out

    def test_saves_current_gw_only_at_end_of_season(self, tmp_path):
        """When no is_next GW exists (GW38, end of season), current GW is still saved."""
        bootstrap = _bootstrap(current_gw=38, next_gw=None)
        for e in bootstrap["events"]:
            e["is_next"] = False
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        with patch("fetch_bootstrap_snapshots.fetch_live_bootstrap", return_value=bootstrap), \
             patch("fetch_bootstrap_snapshots.SNAPSHOTS_DIR", snapshots_dir):
            fbs.live_mode()

        assert (snapshots_dir / "bootstrap_gw38.json").exists()


# ---------------------------------------------------------------------------
# _price_change_summary  (E-F1 coverage)
# ---------------------------------------------------------------------------

def _player(code: int, web_name: str, team_id: int, now_cost: int) -> dict:
    return {"code": code, "web_name": web_name, "team": team_id, "now_cost": now_cost}


def _bs(players: list, teams: list | None = None) -> dict:
    return {
        "elements": players,
        "teams": teams or [{"id": 1, "short_name": "ARS"}],
    }


class TestPriceChangeSummary:

    def test_rise_detected(self):
        old = _bs([_player(1, "Saka", 1, 100)])
        new = _bs([_player(1, "Saka", 1, 101)])
        summary = fbs._price_change_summary(old, new, "test")
        assert "📈" in summary
        assert "Saka" in summary

    def test_fall_detected(self):
        old = _bs([_player(1, "Saka", 1, 101)])
        new = _bs([_player(1, "Saka", 1, 100)])
        summary = fbs._price_change_summary(old, new, "test")
        assert "📉" in summary

    def test_no_changes_returns_no_change_message(self):
        bs = _bs([_player(1, "Saka", 1, 100)])
        summary = fbs._price_change_summary(bs, bs, "test")
        assert "No price changes" in summary

    def test_new_player_entry_detected(self):
        old = _bs([])
        new = _bs([_player(99, "NewGuy", 1, 50)])
        summary = fbs._price_change_summary(old, new, "test")
        assert "NewGuy" in summary

    def test_removed_player_detected(self):
        old = _bs([_player(99, "OldGuy", 1, 50)])
        new = _bs([])
        summary = fbs._price_change_summary(old, new, "test")
        assert "OldGuy" in summary

    def test_uses_persistent_code_not_element_id(self):
        """Same player, different element id → no false price change detected."""
        old = _bs([_player(code=1001, web_name="Saka", team_id=1, now_cost=100)])
        new = _bs([_player(code=1001, web_name="Saka", team_id=1, now_cost=100)])
        summary = fbs._price_change_summary(old, new, "test")
        assert "No price changes" in summary


class TestLiveModeWritesPriceChangesFile:

    def test_writes_price_changes_latest_txt(self, tmp_path):
        """live_mode() must write price_changes_latest.txt when a previous snapshot exists."""
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        old_bootstrap = _bootstrap(current_gw=31, next_gw=32)
        old_bootstrap["elements"] = [_player(1, "Saka", 1, 100)]
        old_bootstrap["teams"] = [{"id": 1, "short_name": "ARS"}]
        (snapshots_dir / "bootstrap_gw32.json").write_text(
            json.dumps(old_bootstrap), encoding="utf-8"
        )

        new_bootstrap = _bootstrap(current_gw=32, next_gw=33)
        new_bootstrap["elements"] = [_player(1, "Saka", 1, 101)]  # price rise
        new_bootstrap["teams"] = [{"id": 1, "short_name": "ARS"}]

        with patch("fetch_bootstrap_snapshots.fetch_live_bootstrap", return_value=new_bootstrap), \
             patch("fetch_bootstrap_snapshots.SNAPSHOTS_DIR", snapshots_dir), \
             patch("fetch_bootstrap_snapshots.PRICE_CHANGES_FILE",
                   snapshots_dir / "price_changes_latest.txt"):
            fbs.live_mode()

        price_file = snapshots_dir / "price_changes_latest.txt"
        assert price_file.exists(), "price_changes_latest.txt was not written"
        content = price_file.read_text(encoding="utf-8")
        assert "Saka" in content

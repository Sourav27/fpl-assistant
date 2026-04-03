"""Integration replay tests: pipeline correctness verified against real historical GW data.

These tests replay real GWs using archived FPL API data (bootstrap-static from
web.archive.org) to catch bugs that unit tests with mock data cannot detect.
Each test was written when a real-GW bug was discovered — before the fix — so the
fix is verified against real data and the bug cannot regress.

Data sources (tests/fixtures/gw{N}/):
  bootstrap.json      — FPL bootstrap-static, pre-deadline snapshot (Wayback Machine)
  bootstrap_post.json — FPL bootstrap-static, post-GW snapshot (Wayback Machine)
  live.json           — Per-player GW scores rebuilt from bootstrap_post.event_points
  entry_picks.json    — User entry picks for the GW (FPL API entry/{id}/event/{gw}/picks/)
  predictions_raw.csv — Raw ML predictions (pre-correction) from results/predictions_gw{N}.csv
  recommend.csv       — Transfer recommendations from results/recommend_gw{N}.csv

GW31 bugs found (2026-03-29):
  A-F4: Semenyo (Man City blank) had xP=12.04 — no blank GW zeroing in pipeline
  A-F1: your_pts=44 logged (correct=57) — captain multiplier not applied, bench included
  A-F3: Walker (GW32 transfer) leaked into GW31 recommended squad comparison
"""
import pytest
import pandas as pd


class TestGW31Replay:
    """Replay tests for GW31 — 4 blank teams (Arsenal, Crystal Palace, Man City, Wolves)."""

    def test_blank_xp_zeroed(self, gw31_fixtures):
        """A-F4: Semenyo (Man City blank) must have corrected xP == 0 but raw_xP > 0."""
        from src.pipeline.predict import apply_xp_corrections

        corrected = apply_xp_corrections(
            gw31_fixtures["predictions_raw"],
            gw31_fixtures["bootstrap"],
            target_gw=31,
        )

        assert "raw_xP" in corrected.columns, "apply_xp_corrections must add raw_xP column"
        assert "xP" in corrected.columns, "apply_xp_corrections must retain xP (corrected)"

        semenyo = corrected[corrected["name"] == "Semenyo"]
        assert len(semenyo) == 1, "Semenyo must be in predictions"
        assert semenyo["xP"].iloc[0] == 0.0, (
            f"Semenyo (Man City blank GW31) must have xP=0, got {semenyo['xP'].iloc[0]}"
        )
        assert semenyo["raw_xP"].iloc[0] > 0.0, (
            "Model raw_xP must be > 0 — model was blind to blank (confirms bug is real)"
        )

    def test_all_blank_team_players_zeroed(self, gw31_fixtures):
        """A-F4: All players from all 4 blank teams must have xP == 0 after correction."""
        from src.pipeline.predict import apply_xp_corrections

        corrected = apply_xp_corrections(
            gw31_fixtures["predictions_raw"],
            gw31_fixtures["bootstrap"],
            target_gw=31,
        )

        blank_teams = {"Arsenal", "Crystal Palace", "Man City", "Wolves"}
        blank_players = corrected[corrected["team"].isin(blank_teams)]

        assert len(blank_players) > 0, "Must have players from blank teams in predictions"
        non_zero = blank_players[blank_players["xP"] != 0.0]
        assert len(non_zero) == 0, (
            f"All blank-team players must have xP=0. "
            f"Non-zero: {non_zero[['name', 'team', 'xP']].to_dict('records')}"
        )

    def test_your_pts_from_api(self, gw31_fixtures):
        """A-F1: your_pts must come from entry_history.points (57), not manual squad sum (44)."""
        from src.pipeline.run import _score_from_entry_picks

        your_pts = _score_from_entry_picks(gw31_fixtures["entry_picks"])
        assert your_pts == 57, (
            f"Expected 57 (B.Fernandes 13pts × captain multiplier 2 = 26, "
            f"correct XI sum = 57), got {your_pts}"
        )

    def test_no_future_gw_transfer_leakage(self, gw31_fixtures):
        """A-F3: GW31 recommended squad must not include Walker (GW32 transfer)."""
        from src.pipeline.run import _filter_gw_transfers

        gw31_rec = _filter_gw_transfers(gw31_fixtures["recommend_df"], current_gw=31)
        players_in = gw31_rec["player_in"].dropna().tolist()

        assert "Walker" not in players_in, (
            "Walker was recommended for GW32, not GW31 — must not appear in "
            "GW31 post-match recommended squad comparison"
        )

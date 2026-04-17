"""End-to-end pipeline tests: verifies that phases chain correctly.

These tests exercise the full data flow (pre-deadline → predict) using mocked
API calls but real in-memory data flow — catching bugs that unit tests on
individual phases cannot detect (e.g. snapshot written by phase 1 not read by
phase 2, optimizer receiving wrong player counts, fallback path silently broken).

All tests run without a vaastav clone (CI path): ep_next from the bootstrap
snapshot is the xP source.
"""
import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _player(code, pid, name, team, pos_type, cost, ep_next, status="a"):
    return {
        "id": pid, "code": code,
        "first_name": name.split()[0], "second_name": name.split()[-1],
        "web_name": name,
        "team": team, "element_type": pos_type,
        "now_cost": cost,
        "total_points": 80, "minutes": 2700,
        "ep_this": str(ep_next - 0.2), "ep_next": str(ep_next),
        "status": status, "chance_of_playing_next_round": None,
        "news": "", "news_added": None,
        "form": "5.0", "selected_by_percent": "10.0",
        "goals_scored": 0, "assists": 0, "clean_sheets": 5,
        "expected_goals": "0.5", "expected_assists": "0.5",
    }


@pytest.fixture
def e2e_bootstrap():
    """Minimal bootstrap with enough players for the optimizer (15-player squad).

    Squad composition:
      2 GK  (teams 1, 2)
      5 DEF (teams 3, 4, 5, 1, 2)
      5 MID (teams 3, 4, 5, 6, 7)
      3 FWD (teams 1, 6, 7)
    Team counts: t1=3, t2=2, t3=2, t4=2, t5=2, t6=2, t7=2  (all ≤ 3 ✓)
    Total cost: 946 (≤ 1000 budget ✓)
    """
    # (code, id, name, team, element_type, cost, ep_next)
    specs = [
        # GK
        (1001, 1, "GK One",   1, 1, 55, 4.5),
        (1002, 2, "GK Two",   2, 1, 48, 3.8),
        # DEF
        (2001, 3, "DEF One",  3, 2, 55, 5.0),
        (2002, 4, "DEF Two",  4, 2, 50, 4.8),
        (2003, 5, "DEF Three",5, 2, 50, 4.6),
        (2004, 6, "DEF Four", 1, 2, 52, 5.2),
        (2005, 7, "DEF Five", 2, 2, 53, 4.9),
        # MID
        (3001, 8,  "MID One",   3, 3, 75, 6.8),
        (3002, 9,  "MID Two",   4, 3, 70, 6.5),
        (3003, 10, "MID Three", 5, 3, 65, 6.0),
        (3004, 11, "MID Four",  6, 3, 60, 5.8),
        (3005, 12, "MID Five",  7, 3, 58, 5.5),
        # FWD
        (4001, 13, "FWD One",   1, 4, 90, 7.5),
        (4002, 14, "FWD Two",   6, 4, 85, 7.0),
        (4003, 15, "FWD Three", 7, 4, 80, 6.5),
    ]
    players = [_player(*s) for s in specs]

    return {
        "events": [
            {
                "id": 30,
                "deadline_time": "2026-03-14T11:00:00Z",
                "is_current": True, "is_next": False, "finished": True,
            },
            {
                "id": 31,
                "deadline_time": "2026-03-20T18:30:00Z",
                "is_current": False, "is_next": True, "finished": False,
            },
        ],
        "elements": players,
        "teams": [
            {"id": i, "name": f"Team{i}", "short_name": f"T{i}", "code": i * 10}
            for i in range(1, 8)
        ],
        "element_types": [
            {"id": 1, "singular_name": "Goalkeeper",  "singular_name_short": "GKP", "plural_name_short": "GKP"},
            {"id": 2, "singular_name": "Defender",    "singular_name_short": "DEF", "plural_name_short": "DEF"},
            {"id": 3, "singular_name": "Midfielder",  "singular_name_short": "MID", "plural_name_short": "MID"},
            {"id": 4, "singular_name": "Forward",     "singular_name_short": "FWD", "plural_name_short": "FWD"},
        ],
    }


# ---------------------------------------------------------------------------
# E2E: pre-deadline → predict (the "full" command path)
# ---------------------------------------------------------------------------

class TestPipelineE2E:
    """Chains phase_pre_deadline → phase_predict with no vaastav data (CI path)."""

    def _run_full(self, tmp_path, e2e_bootstrap):
        """Helper: run both phases against tmp_path, return phase_predict result."""
        import src.pipeline.run as run_mod

        snapshot_dir = tmp_path / "snapshots"
        gw_dir = tmp_path / "FPL" / "data" / "2025-26" / "gws"

        with patch("src.pipeline.run.fetch_bootstrap", return_value=e2e_bootstrap), \
             patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.RESULTS_DIR", tmp_path / "results"), \
             patch("src.pipeline.run.SNAPSHOTS_DIR", snapshot_dir), \
             patch("src.pipeline.run.ACTIVE_MODEL", tmp_path / "nonexistent.sav"), \
             patch("src.pipeline.run.ACTIVE_MODELS", {}), \
             patch("src.pipeline.run.CURRENT_SEASON", "2025-26"):

            snapshot_dir.mkdir(parents=True)
            gw_dir.mkdir(parents=True)
            (tmp_path / "results").mkdir(parents=True)

            next_gw = run_mod.phase_pre_deadline()
            assert next_gw == 31, f"Expected next_gw=31, got {next_gw}"

            result = run_mod.phase_predict(target_gw=next_gw)

        return result, tmp_path / "results"

    def test_pre_deadline_writes_bootstrap_snapshot(self, tmp_path, e2e_bootstrap):
        """Phase 1 must write bootstrap_gw{N}.json to SNAPSHOTS_DIR."""
        import src.pipeline.run as run_mod

        snapshot_dir = tmp_path / "snapshots"
        with patch("src.pipeline.run.fetch_bootstrap", return_value=e2e_bootstrap), \
             patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.RESULTS_DIR", tmp_path / "results"), \
             patch("src.pipeline.run.SNAPSHOTS_DIR", snapshot_dir), \
             patch("src.pipeline.run.CURRENT_SEASON", "2025-26"):
            (tmp_path / "FPL" / "data" / "2025-26" / "gws").mkdir(parents=True)
            (tmp_path / "results").mkdir()
            snapshot_dir.mkdir()
            run_mod.phase_pre_deadline()

        assert (snapshot_dir / "bootstrap_gw31.json").exists()
        data = json.loads((snapshot_dir / "bootstrap_gw31.json").read_text())
        assert "elements" in data

    def test_pre_deadline_writes_xp_snapshot(self, tmp_path, e2e_bootstrap):
        """Phase 1 must write xP{N}.csv to the vaastav gw dir."""
        import src.pipeline.run as run_mod

        snapshot_dir = tmp_path / "snapshots"
        gw_dir = tmp_path / "FPL" / "data" / "2025-26" / "gws"
        with patch("src.pipeline.run.fetch_bootstrap", return_value=e2e_bootstrap), \
             patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.RESULTS_DIR", tmp_path / "results"), \
             patch("src.pipeline.run.SNAPSHOTS_DIR", snapshot_dir), \
             patch("src.pipeline.run.CURRENT_SEASON", "2025-26"):
            gw_dir.mkdir(parents=True)
            (tmp_path / "results").mkdir()
            snapshot_dir.mkdir()
            run_mod.phase_pre_deadline()

        xp_path = gw_dir / "xP31.csv"
        assert xp_path.exists()
        xp_df = pd.read_csv(xp_path)
        assert set(xp_df.columns) >= {"id", "xP"}
        assert len(xp_df) == 15

    def test_predict_ep_next_fallback_writes_xi_and_squad(self, tmp_path, e2e_bootstrap):
        """Phase 2 (ep_next fallback, no vaastav) must write xi_gw{N}.csv and squad_gw{N}.csv."""
        result, results_dir = self._run_full(tmp_path, e2e_bootstrap)

        assert (results_dir / "xi_gw31.csv").exists(), "xi_gw31.csv not written"
        assert (results_dir / "squad_gw31.csv").exists(), "squad_gw31.csv not written"

    def test_predict_ep_next_fallback_writes_predictions(self, tmp_path, e2e_bootstrap):
        """Phase 2 must write predictions_gw{N}.csv (input for recommend phase)."""
        _, results_dir = self._run_full(tmp_path, e2e_bootstrap)

        assert (results_dir / "predictions_gw31.csv").exists(), "predictions_gw31.csv not written"

    def test_predict_squad_has_15_players(self, tmp_path, e2e_bootstrap):
        """The selected squad must always contain exactly 15 players."""
        result, results_dir = self._run_full(tmp_path, e2e_bootstrap)

        squad = pd.read_csv(results_dir / "squad_gw31.csv")
        assert len(squad) == 15, f"Expected 15-player squad, got {len(squad)}"

    def test_predict_xi_has_11_players(self, tmp_path, e2e_bootstrap):
        """The starting XI must always contain exactly 11 players."""
        result, results_dir = self._run_full(tmp_path, e2e_bootstrap)

        xi = pd.read_csv(results_dir / "xi_gw31.csv")
        assert len(xi) == 11, f"Expected 11-player XI, got {len(xi)}"

    def test_predict_xi_has_exactly_one_gk(self, tmp_path, e2e_bootstrap):
        """Starting XI must contain exactly 1 GK."""
        result, results_dir = self._run_full(tmp_path, e2e_bootstrap)

        xi = pd.read_csv(results_dir / "xi_gw31.csv")
        assert (xi["position"] == "GK").sum() == 1

    def test_predict_xi_valid_formation(self, tmp_path, e2e_bootstrap):
        """XI must satisfy FPL formation rules: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD."""
        result, results_dir = self._run_full(tmp_path, e2e_bootstrap)

        xi = pd.read_csv(results_dir / "xi_gw31.csv")
        pos = xi["position"].value_counts()
        assert pos.get("GK", 0) == 1
        assert 3 <= pos.get("DEF", 0) <= 5
        assert 2 <= pos.get("MID", 0) <= 5
        assert 1 <= pos.get("FWD", 0) <= 3

    def test_predict_max_3_from_same_club(self, tmp_path, e2e_bootstrap):
        """Squad must not contain more than 3 players from any single club."""
        result, results_dir = self._run_full(tmp_path, e2e_bootstrap)

        squad = pd.read_csv(results_dir / "squad_gw31.csv")
        max_per_team = squad["team"].value_counts().max()
        assert max_per_team <= 3, f"Club limit violated: max {max_per_team} from one club"

    def test_predict_snapshot_read_from_phase1_output(self, tmp_path, e2e_bootstrap):
        """Phase 2 must use the snapshot written by phase 1 (tests phase chaining)."""
        result, results_dir = self._run_full(tmp_path, e2e_bootstrap)

        # If phase chaining is broken (phase 2 can't find snapshot), it either
        # fails or writes an empty squad. A non-empty squad proves the snapshot
        # written by phase 1 was successfully consumed by phase 2.
        squad = pd.read_csv(results_dir / "squad_gw31.csv")
        assert len(squad) > 0, "Empty squad — phase 2 likely failed to read phase 1 snapshot"

    def test_full_pipeline_budget_constraint(self, tmp_path, e2e_bootstrap):
        """Selected squad must cost ≤ £100M (1000 in FPL tenths)."""
        result, results_dir = self._run_full(tmp_path, e2e_bootstrap)

        squad = pd.read_csv(results_dir / "squad_gw31.csv")
        total_cost = squad["now_cost"].sum()
        assert total_cost <= 1000, f"Budget exceeded: {total_cost / 10:.1f}M > 100M"

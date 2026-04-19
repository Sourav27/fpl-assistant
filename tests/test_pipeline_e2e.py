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

    def _gw31_dir(self, results_dir):
        return results_dir / "2025-26" / "gw31"

    def test_predict_ep_next_fallback_writes_xi_and_squad(self, tmp_path, e2e_bootstrap):
        """Phase 2 (ep_next fallback, no vaastav) must write xi.csv and optimal_squad.csv."""
        result, results_dir = self._run_full(tmp_path, e2e_bootstrap)
        gw31 = self._gw31_dir(results_dir)

        assert (gw31 / "xi.csv").exists(), "xi.csv not written"
        assert (gw31 / "optimal_squad.csv").exists(), "optimal_squad.csv not written"

    def test_predict_ep_next_fallback_writes_predictions(self, tmp_path, e2e_bootstrap):
        """Phase 2 must write predictions.csv (input for recommend phase)."""
        _, results_dir = self._run_full(tmp_path, e2e_bootstrap)
        gw31 = self._gw31_dir(results_dir)

        assert (gw31 / "predictions.csv").exists(), "predictions.csv not written"

    def test_predict_squad_has_15_players(self, tmp_path, e2e_bootstrap):
        """The selected squad must always contain exactly 15 players."""
        result, results_dir = self._run_full(tmp_path, e2e_bootstrap)
        gw31 = self._gw31_dir(results_dir)

        squad = pd.read_csv(gw31 / "optimal_squad.csv")
        assert len(squad) == 15, f"Expected 15-player squad, got {len(squad)}"

    def test_predict_xi_has_11_players(self, tmp_path, e2e_bootstrap):
        """The starting XI must always contain exactly 11 players."""
        result, results_dir = self._run_full(tmp_path, e2e_bootstrap)
        gw31 = self._gw31_dir(results_dir)

        xi = pd.read_csv(gw31 / "xi.csv")
        assert len(xi) == 11, f"Expected 11-player XI, got {len(xi)}"

    def test_predict_xi_has_exactly_one_gk(self, tmp_path, e2e_bootstrap):
        """Starting XI must contain exactly 1 GK."""
        result, results_dir = self._run_full(tmp_path, e2e_bootstrap)
        gw31 = self._gw31_dir(results_dir)

        xi = pd.read_csv(gw31 / "xi.csv")
        assert (xi["position"] == "GK").sum() == 1

    def test_predict_xi_valid_formation(self, tmp_path, e2e_bootstrap):
        """XI must satisfy FPL formation rules: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD."""
        result, results_dir = self._run_full(tmp_path, e2e_bootstrap)
        gw31 = self._gw31_dir(results_dir)

        xi = pd.read_csv(gw31 / "xi.csv")
        pos = xi["position"].value_counts()
        assert pos.get("GK", 0) == 1
        assert 3 <= pos.get("DEF", 0) <= 5
        assert 2 <= pos.get("MID", 0) <= 5
        assert 1 <= pos.get("FWD", 0) <= 3

    def test_predict_max_3_from_same_club(self, tmp_path, e2e_bootstrap):
        """Squad must not contain more than 3 players from any single club."""
        result, results_dir = self._run_full(tmp_path, e2e_bootstrap)
        gw31 = self._gw31_dir(results_dir)

        squad = pd.read_csv(gw31 / "optimal_squad.csv")
        max_per_team = squad["team"].value_counts().max()
        assert max_per_team <= 3, f"Club limit violated: max {max_per_team} from one club"

    def test_predict_snapshot_read_from_phase1_output(self, tmp_path, e2e_bootstrap):
        """Phase 2 must use the snapshot written by phase 1 (tests phase chaining)."""
        result, results_dir = self._run_full(tmp_path, e2e_bootstrap)
        gw31 = self._gw31_dir(results_dir)

        # If phase chaining is broken (phase 2 can't find snapshot), it either
        # fails or writes an empty squad. A non-empty squad proves the snapshot
        # written by phase 1 was successfully consumed by phase 2.
        squad = pd.read_csv(gw31 / "optimal_squad.csv")
        assert len(squad) > 0, "Empty squad — phase 2 likely failed to read phase 1 snapshot"

    def test_full_pipeline_budget_constraint(self, tmp_path, e2e_bootstrap):
        """Selected squad must cost ≤ £100M (1000 in FPL tenths)."""
        result, results_dir = self._run_full(tmp_path, e2e_bootstrap)
        gw31 = self._gw31_dir(results_dir)

        squad = pd.read_csv(gw31 / "optimal_squad.csv")
        total_cost = squad["now_cost"].sum()
        assert total_cost <= 1000, f"Budget exceeded: {total_cost / 10:.1f}M > 100M"


# ---------------------------------------------------------------------------
# E2E: partial manifest (only MID model available) — the CI failure pattern
# ---------------------------------------------------------------------------

class TestPartialManifestFallback:
    """Tests the scenario that caused the GW33 CI failure:
    active_models.json only lists a subset of positions (e.g. only MID was
    promoted). predict_next_gw_per_position only iterates over listed positions,
    leaving GK/DEF/FWD players unprocessed. The phase_predict fallback must
    fill in the missing players from ep_next so the optimizer receives a full
    15-player-eligible pool.
    """

    def _run_predict_with_mid_only_model(self, tmp_path, e2e_bootstrap):
        """Run phase_predict with a fake MID-only model (other positions have no model)."""
        import src.pipeline.run as run_mod
        from unittest.mock import MagicMock

        snapshot_dir = tmp_path / "snapshots"
        gw_dir = tmp_path / "FPL" / "data" / "2025-26" / "gws"
        results_dir = tmp_path / "results"

        snapshot_dir.mkdir(parents=True)
        gw_dir.mkdir(parents=True)
        results_dir.mkdir(parents=True)

        # Write bootstrap snapshot (normally written by phase_pre_deadline)
        import json
        (snapshot_dir / "bootstrap_gw31.json").write_text(json.dumps(e2e_bootstrap))

        # Fake model that returns zeros for any input
        fake_mid_model = MagicMock()
        fake_mid_model.predict.return_value = [0.0]

        # Manifest only has MID — simulates a partial promotion cycle
        partial_models = {"MID": fake_mid_model}

        with patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.RESULTS_DIR", results_dir), \
             patch("src.pipeline.run.SNAPSHOTS_DIR", snapshot_dir), \
             patch("src.pipeline.run.ACTIVE_MODEL", tmp_path / "nonexistent.sav"), \
             patch("src.pipeline.run.ACTIVE_MODELS", {"MID": tmp_path / "fake_mid.sav"}), \
             patch("src.pipeline.run.CURRENT_SEASON", "2025-26"), \
             patch("src.pipeline.run.load_position_models", return_value=partial_models):
            run_mod.phase_predict(target_gw=31)

        return results_dir

    def _gw31_dir(self, results_dir):
        return results_dir / "2025-26" / "gw31"

    def test_partial_manifest_still_produces_full_squad(self, tmp_path, e2e_bootstrap):
        """When only MID model is in manifest, fallback must fill GK/DEF/FWD from ep_next."""
        results_dir = self._run_predict_with_mid_only_model(tmp_path, e2e_bootstrap)
        gw31 = self._gw31_dir(results_dir)

        squad = pd.read_csv(gw31 / "optimal_squad.csv")
        assert len(squad) == 15, (
            f"Partial manifest fallback must produce 15-player squad, got {len(squad)}"
        )

    def test_partial_manifest_predictions_has_all_positions(self, tmp_path, e2e_bootstrap):
        """predictions.csv must include GK/DEF/MID/FWD even if only MID model exists."""
        results_dir = self._run_predict_with_mid_only_model(tmp_path, e2e_bootstrap)
        gw31 = self._gw31_dir(results_dir)

        preds = pd.read_csv(gw31 / "predictions.csv")
        assert len(preds) > 0, "Predictions must not be empty when partial manifest is used"
        positions_present = set(preds["position"].unique())
        assert positions_present >= {"GK", "DEF", "MID", "FWD"}, (
            f"All positions must be present in predictions, got: {positions_present}"
        )

    def test_partial_manifest_xi_valid(self, tmp_path, e2e_bootstrap):
        """XI must be valid even when only one position model exists."""
        results_dir = self._run_predict_with_mid_only_model(tmp_path, e2e_bootstrap)
        gw31 = self._gw31_dir(results_dir)

        xi = pd.read_csv(gw31 / "xi.csv")
        assert len(xi) == 11, f"XI must have 11 players, got {len(xi)}"
        assert (xi["position"] == "GK").sum() == 1


# ---------------------------------------------------------------------------
# E2E: recommend must exit gracefully on empty predictions
# ---------------------------------------------------------------------------

class TestRecommendEmptyPredictions:
    """Tests that phase_recommend exits cleanly (no crash) when predictions
    file exists but contains 0 rows — the downstream half of the GW33 failure.
    """

    def test_recommend_returns_none_on_empty_predictions(self, tmp_path, monkeypatch):
        """phase_recommend must return None, not crash, when predictions CSV is empty."""
        import src.pipeline.run as run_mod

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Write an empty predictions CSV (headers only) at the new gw_dir path
        gw33_dir = results_dir / "2025-26" / "gw33"
        gw33_dir.mkdir(parents=True, exist_ok=True)
        (gw33_dir / "predictions.csv").write_text(
            "element,code,name,position,team,xP,now_cost\n"
        )

        monkeypatch.setattr(run_mod, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(run_mod, "load_user_config", lambda: {
            "teams": {"default": {"entry_id": 123}},
            "preferences": {"horizon_gws": 1, "fdr_sensitivity": 0.15, "max_hit_points": 8},
        })

        result = run_mod.phase_recommend(target_gw=33, team_key="default")
        assert result is None, "phase_recommend must return None on empty predictions, not crash"

    def test_optimize_team_raises_on_empty(self):
        """optimize_team must raise ValueError, not IndexError, when given empty players."""
        from src.pipeline.optimize import optimize_team

        empty = pd.DataFrame(columns=["element", "name", "position", "team", "xP", "now_cost"])
        with pytest.raises(ValueError, match="empty"):
            optimize_team(empty)


# ---------------------------------------------------------------------------
# E2E: post-gw accuracy log + Discord notification
# ---------------------------------------------------------------------------

class TestPostGwDiscord:
    """Tests that phase_post_gw writes a correct accuracy log and fires Discord.

    These tests use a finished GW31 bootstrap (finished=True on event id=31
    — note: post-gw runs on the CURRENT finished GW, not next).
    """

    def _bootstrap_finished(self, e2e_bootstrap):
        """Return a bootstrap where GW31 is current+finished, GW32 is next."""
        import copy
        bs = copy.deepcopy(e2e_bootstrap)
        bs["events"] = [
            {
                "id": 31,
                "deadline_time": "2026-03-20T18:30:00Z",
                "is_current": True, "is_next": False, "finished": True,
            },
            {
                "id": 32,
                "deadline_time": "2026-03-27T18:30:00Z",
                "is_current": False, "is_next": True, "finished": False,
            },
        ]
        return bs

    def _live_df(self, e2e_bootstrap):
        """Fake live GW data — assign varying actual_points so spearman_rho is defined."""
        players = e2e_bootstrap["elements"]
        rows = [
            {"element": p["id"], "total_points": (i % 5) + 2, "name": p["web_name"]}
            for i, p in enumerate(players)
        ]
        return pd.DataFrame(rows)

    def _setup_results(self, tmp_path, e2e_bootstrap):
        """Write predictions, squad, and squad_recommend CSVs to results dir."""
        results = tmp_path / "results"
        results.mkdir(parents=True, exist_ok=True)

        players = e2e_bootstrap["elements"]
        from src.pipeline.fetch import ELEMENT_TYPE_MAP
        rows = [{
            "element": p["id"], "code": p["code"],
            "name": p["web_name"], "xP": float(p["ep_next"]),
            "now_cost": p["now_cost"],
            "position": ELEMENT_TYPE_MAP.get(p["element_type"], "MID"),
            "team": f"Team{p['team']}",
        } for p in players]
        df = pd.DataFrame(rows)

        gw31 = results / "2025-26" / "gw31"
        gw31.mkdir(parents=True, exist_ok=True)
        df.to_csv(gw31 / "predictions.csv", index=False)
        df.to_csv(gw31 / "optimal_squad.csv", index=False)
        df.head(15).to_csv(gw31 / "squad_recommend.csv", index=False)
        return results

    def test_post_gw_accuracy_log_has_spearman_rho(self, tmp_path, e2e_bootstrap):
        """phase_post_gw must write a non-null spearman_rho when picks data is available."""
        import src.pipeline.run as run_mod
        from unittest.mock import MagicMock

        bs_finished = self._bootstrap_finished(e2e_bootstrap)
        live_df = self._live_df(e2e_bootstrap)
        results = self._setup_results(tmp_path, e2e_bootstrap)

        entry_picks_response = MagicMock()
        entry_picks_response.json.return_value = {
            "entry_history": {"points": 55},
            "picks": [{"element": p["id"]} for p in e2e_bootstrap["elements"][:11]],
        }
        entry_response = MagicMock()
        entry_response.json.return_value = {"leagues": {"classic": []}}

        with patch("src.pipeline.run.fetch_bootstrap", return_value=bs_finished), \
             patch("src.pipeline.run.fetch_fixtures", return_value=[]), \
             patch("src.pipeline.run.fetch_live_gw_data", return_value=live_df), \
             patch("src.pipeline.run.RESULTS_DIR", results), \
             patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.load_user_config", return_value={
                 "teams": {"default": {"entry_id": 123}},
                 "preferences": {},
             }), \
             patch("src.pipeline.run._api_get_with_retry", side_effect=[
                 entry_picks_response, entry_response,
                 MagicMock(**{"json.return_value": {"current": [{"event": 31, "percentile_rank": 25}]}}),
             ]):
            run_mod.phase_post_gw()

        log = pd.read_csv(results / "accuracy_log.csv")
        assert (log["spearman_rho"].notna()).any(), "spearman_rho must be written"

    def test_post_gw_accuracy_log_has_wildcard_pts(self, tmp_path, e2e_bootstrap):
        """phase_post_gw must write wildcard_pts when squad_gw{N}.csv exists."""
        import src.pipeline.run as run_mod
        from unittest.mock import MagicMock

        bs_finished = self._bootstrap_finished(e2e_bootstrap)
        live_df = self._live_df(e2e_bootstrap)
        results = self._setup_results(tmp_path, e2e_bootstrap)

        entry_picks_response = MagicMock()
        entry_picks_response.json.return_value = {
            "entry_history": {"points": 55},
            "picks": [{"element": p["id"]} for p in e2e_bootstrap["elements"][:11]],
        }

        with patch("src.pipeline.run.fetch_bootstrap", return_value=bs_finished), \
             patch("src.pipeline.run.fetch_fixtures", return_value=[]), \
             patch("src.pipeline.run.fetch_live_gw_data", return_value=live_df), \
             patch("src.pipeline.run.RESULTS_DIR", results), \
             patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.load_user_config", return_value={
                 "teams": {"default": {"entry_id": 123}},
                 "preferences": {},
             }), \
             patch("src.pipeline.run._api_get_with_retry", side_effect=[
                 entry_picks_response,
                 MagicMock(**{"json.return_value": {"leagues": {"classic": []}}}),
                 MagicMock(**{"json.return_value": {"current": []}}),
             ]):
            run_mod.phase_post_gw()

        log = pd.read_csv(results / "accuracy_log.csv")
        assert (log["wildcard_pts"].notna()).any(), "wildcard_pts must be written when squad CSV exists"
        assert log.iloc[-1]["wildcard_pts"] > 0

    def test_post_gw_discord_called(self, tmp_path, e2e_bootstrap):
        """format_accuracy_discord.py must produce non-empty output for a complete log row."""
        import subprocess, sys, json as _json
        from pathlib import Path as _Path

        results = tmp_path / "results"
        results.mkdir(parents=True, exist_ok=True)
        log_path = results / "accuracy_log.csv"
        log_path.write_text(
            "gw,your_pts,your_predicted_xp,recommended_pts,recommended_xp,"
            "wildcard_pts,wildcard_xp,dream_team_pts,your_percentile_rank,"
            "best_score,top_1k_score,top_10k_score,top_100k_score,top_1m_score,"
            "avg_score,median_score,ranked_count,spearman_rho,timestamp\n"
            "31,55,48.5,60,52.0,70,65.0,129,25,109,66,50,38,,38,,12914049,0.65,2026-04-17T00:00:00+00:00\n"
        )

        repo_root = _Path(__file__).parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/format_accuracy_discord.py", str(log_path), "31"],
            capture_output=True, text=True, encoding="utf-8",
            cwd=repo_root,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "GW31" in result.stdout
        assert "55" in result.stdout
        assert "0.650" in result.stdout
        assert "70" in result.stdout

    def test_check_gw_finished_script(self, tmp_path, e2e_bootstrap):
        """check_gw_finished.py must detect finished=True when current GW is done."""
        import subprocess, sys, json as _json
        from pathlib import Path as _Path

        bs = dict(e2e_bootstrap)
        bs["events"] = [
            {"id": 31, "deadline_time": "2026-03-20T18:30:00Z",
             "is_current": True, "is_next": False, "finished": True},
        ]
        snap = tmp_path / "bootstrap_gw31.json"
        snap.write_text(_json.dumps(bs))

        repo_root = _Path(__file__).parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/check_gw_finished.py", str(snap)],
            capture_output=True, text=True,
            cwd=repo_root,
        )
        assert result.returncode == 0
        assert "Finished: True" in result.stdout

    def test_check_gw_not_finished(self, tmp_path, e2e_bootstrap):
        """check_gw_finished.py must detect finished=False when GW is ongoing."""
        import subprocess, sys, json as _json
        from pathlib import Path as _Path

        bs = dict(e2e_bootstrap)
        bs["events"] = [
            {"id": 31, "deadline_time": "2026-03-20T18:30:00Z",
             "is_current": True, "is_next": False, "finished": False},
        ]
        snap = tmp_path / "bootstrap_gw31.json"
        snap.write_text(_json.dumps(bs))

        repo_root = _Path(__file__).parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/check_gw_finished.py", str(snap)],
            capture_output=True, text=True,
            cwd=repo_root,
        )
        assert result.returncode == 0
        assert "Finished: False" in result.stdout

# tests/test_run.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
from src.pipeline.run import phase_pre_deadline, phase_predict, phase_post_gw, phase_retrain
from src.config import UserConfigError


class TestPhasePreDeadline:
    def test_saves_xp_snapshot(self, tmp_path, sample_bootstrap_json):
        with patch("src.pipeline.run.fetch_bootstrap", return_value=sample_bootstrap_json), \
             patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.RESULTS_DIR", tmp_path / "results"):
            (tmp_path / "FPL" / "data" / "2025-26" / "gws").mkdir(parents=True)
            (tmp_path / "results").mkdir()

            next_gw = phase_pre_deadline()

        assert next_gw == 31
        xp_path = tmp_path / "FPL" / "data" / "2025-26" / "gws" / "xP31.csv"
        assert xp_path.exists()

    def test_saves_bootstrap_snapshot(self, tmp_path, sample_bootstrap_json):
        with patch("src.pipeline.run.fetch_bootstrap", return_value=sample_bootstrap_json), \
             patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.RESULTS_DIR", tmp_path / "results"), \
             patch("src.pipeline.run.SNAPSHOTS_DIR", tmp_path / "results" / "snapshots"):
            (tmp_path / "FPL" / "data" / "2025-26" / "gws").mkdir(parents=True)
            (tmp_path / "results").mkdir()
            (tmp_path / "results" / "snapshots").mkdir(parents=True)

            phase_pre_deadline()

        snapshot_path = tmp_path / "results" / "snapshots" / "bootstrap_gw31.json"
        assert snapshot_path.exists()
        data = json.loads(snapshot_path.read_text())
        assert "elements" in data


class TestPhasePostGw:
    def test_saves_live_gw_csv(self, tmp_path, sample_bootstrap_json, sample_player_history_json):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_player_history_json

        gw_dir = tmp_path / "FPL" / "data" / "2025-26" / "gws"
        gw_dir.mkdir(parents=True)

        with patch("src.pipeline.run.fetch_bootstrap", return_value=sample_bootstrap_json), \
             patch("src.pipeline.run.fetch_fixtures", return_value=[]), \
             patch("src.pipeline.run.fetch_live_gw_data") as mock_fetch_live, \
             patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.RESULTS_DIR", tmp_path / "results"):
            import pandas as pd
            mock_fetch_live.return_value = pd.DataFrame({
                "name": ["Saka"], "element": [3], "GW": [30],
                "total_points": [8], "position": ["MID"], "team": ["Arsenal"],
            })
            (tmp_path / "results").mkdir()

            phase_post_gw()

        live_path = gw_dir / "gw30_live.csv"
        assert live_path.exists()


class TestPhasePredict:
    def test_writes_output_csvs(self, tmp_path, sample_bootstrap_json):
        import pandas as pd
        import numpy as np

        # Create minimal vaastav data
        gw_dir = tmp_path / "FPL" / "data" / "2025-26" / "gws"
        gw_dir.mkdir(parents=True)
        rows = []
        for gw in range(1, 10):
            rows.append({
                "name": "Saka", "position": "MID", "team": "Arsenal",
                "element": 3, "total_points": np.random.randint(2, 12),
                "minutes": 90, "goals_scored": 0, "assists": 0,
                "clean_sheets": 0, "ict_index": 10.0, "influence": 30.0,
                "creativity": 25.0, "threat": 40.0, "bps": 20, "bonus": 1,
                "value": 105, "transfers_in": 5000, "transfers_out": 1000,
                "selected": 3000000, "was_home": True, "opponent_team": 10,
                "fixture": gw, "round": gw, "GW": gw,
            })
        pd.DataFrame(rows).to_csv(gw_dir / "merged_gw.csv", index=False)

        # Save bootstrap snapshot for availability filtering
        results_dir = tmp_path / "results"
        snapshot_dir = results_dir / "snapshots"
        snapshot_dir.mkdir(parents=True)
        with open(snapshot_dir / "bootstrap_gw10.json", "w") as f:
            json.dump(sample_bootstrap_json, f)

        # Mock model to not exist → falls back to xP=0
        with patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.RESULTS_DIR", results_dir), \
             patch("src.pipeline.run.ACTIVE_MODEL", tmp_path / "nonexistent.sav"), \
             patch("src.pipeline.run.CURRENT_SEASON", "2025-26"):
            # Will warn about missing model, use fallback
            result = phase_predict(target_gw=10)

        assert (results_dir / "xi_gw10.csv").exists()
        assert (results_dir / "squad_gw10.csv").exists()


class TestPhaseRetrain:
    def test_saves_new_model(self, tmp_path):
        import pandas as pd
        import numpy as np

        # Create minimal vaastav data with enough rows
        gw_dir = tmp_path / "FPL" / "data" / "2025-26" / "gws"
        gw_dir.mkdir(parents=True)
        rows = []
        # 10 players × 20 GWs = 200 raw rows → ~120 after rolling-8 NaN drop (≥100 threshold)
        for player in range(1, 11):
            for gw in range(1, 21):
                rows.append({
                    "name": f"Player{player}", "position": "MID", "team": "Arsenal",
                    "element": player, "total_points": np.random.randint(0, 15),
                    "minutes": 90, "goals_scored": 0, "assists": 0,
                    "clean_sheets": 0, "ict_index": 10.0, "influence": 30.0,
                    "creativity": 25.0, "threat": 40.0, "bps": 20, "bonus": 1,
                    "value": 100, "transfers_in": 5000, "transfers_out": 1000,
                    "selected": 3000000, "was_home": True, "opponent_team": 10,
                    "fixture": gw, "round": gw, "GW": gw, "season": "2025-26",
                })
        pd.DataFrame(rows).to_csv(gw_dir / "merged_gw.csv", index=False)

        models_dir = tmp_path / "models"
        with patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.MODELS_DIR", models_dir), \
             patch("src.pipeline.run.ACTIVE_MODEL", models_dir / "rf_model.sav"), \
             patch("src.pipeline.run.CURRENT_SEASON", "2025-26"), \
             patch("src.pipeline.promote.run_promotion_pipeline") as mock_promote:
            phase_retrain(target_gw=32)

        # Track I: retrain now saves per-position models with date-based naming.
        # With all test data as "MID" position (100 rows), at least one rf_mid_*.sav must exist.
        saved_models = list(models_dir.glob("rf_mid_*.sav"))
        assert len(saved_models) >= 1, f"Expected at least one per-position model, found: {list(models_dir.iterdir())}"
        # Track I: promotion pipeline must be called with the trained models
        assert mock_promote.called, "run_promotion_pipeline should be called after training"


class TestRecommendPhase:
    def test_recommend_phase_requires_predictions_file(self, tmp_path, monkeypatch):
        """If predictions_gw{N}.csv is missing, phase should print error and return."""
        from src.pipeline.run import phase_recommend
        import src.pipeline.run as run_mod
        monkeypatch.setattr(run_mod, "RESULTS_DIR", tmp_path)
        # No user_config.yaml → should raise or print error cleanly
        result = phase_recommend(target_gw=33, team_key="default")
        assert result is None

    def test_recommend_phase_wildcard_not_auto_detected(self):
        """active_chip is ignored — wildcard mode requires explicit --wildcard flag."""
        from src.pipeline.user import UserTeamState
        from src.pipeline.run import _is_wildcard_mode
        state = UserTeamState(
            entry_id=123, current_squad=list(range(1, 16)),
            squad_codes=list(range(101, 116)),
            selling_prices={i: 67 for i in range(1, 16)},
            bank=0, free_transfers=1, active_chip="wildcard", total_value=0,
        )
        # active_chip='wildcard' no longer auto-activates wildcard mode
        assert _is_wildcard_mode(state, wildcard_flag=False) is False

    def test_recommend_phase_wildcard_flag_overrides(self):
        from src.pipeline.user import UserTeamState
        from src.pipeline.run import _is_wildcard_mode
        state = UserTeamState(
            entry_id=123, current_squad=list(range(1, 16)),
            squad_codes=list(range(101, 116)),
            selling_prices={i: 67 for i in range(1, 16)},
            bank=0, free_transfers=1, active_chip=None, total_value=0,
        )
        assert _is_wildcard_mode(state, wildcard_flag=True) is True
        assert _is_wildcard_mode(state, wildcard_flag=False) is False


class TestPostGwAnalysis:
    def test_post_gw_skips_analysis_when_no_config(self, tmp_path, monkeypatch):
        """If user_config.yaml missing, post-gw still completes (analysis skipped)."""
        import src.pipeline.run as run_mod
        from unittest.mock import patch, MagicMock
        monkeypatch.setattr(run_mod, "RESULTS_DIR", tmp_path)
        # Mock API calls to return minimal data
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "events": [{"id": 30, "is_current": True, "is_next": False,
                        "finished": True, "highest_score": 100,
                        "average_entry_score": 40, "ranked_count": 10000000}],
            "elements": [], "teams": [],
        }
        with patch("src.pipeline.run.fetch_bootstrap", return_value=mock_resp.json()), \
             patch("src.pipeline.run.fetch_fixtures", return_value=[]), \
             patch("src.pipeline.run.fetch_live_gw_data", return_value=pd.DataFrame()), \
             patch("src.pipeline.run.load_user_config", side_effect=UserConfigError("missing")):
            # Should not raise
            run_mod.phase_post_gw()

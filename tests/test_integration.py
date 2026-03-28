# tests/test_integration.py
"""Integration test: runs the full pipeline on real vaastav data (2024-25 season)."""
import pytest
import pandas as pd
from pathlib import Path
from src.config import VAASTAV_DIR
from src.pipeline.prepare import build_merged_dataset
from src.pipeline.features import engineer_features
from src.pipeline.optimize import optimize_team


@pytest.mark.skipif(
    not (VAASTAV_DIR / "data" / "2024-25" / "gws" / "merged_gw.csv").exists(),
    reason="Vaastav dataset not cloned"
)
class TestIntegration:
    def test_full_pipeline_2024_25(self):
        """Run prepare -> features -> optimize on real 2024-25 data."""
        # Prepare
        merged = build_merged_dataset(seasons=["2024-25"])
        assert len(merged) > 5000

        # Features
        features = engineer_features(merged)
        assert len(features) > 1000
        assert "total_points_roll_4" in features.columns

        # Get latest row per player
        latest = features.sort_values(["element", "GW"]).groupby("element").last().reset_index()

        # Build optimizer input (use actual total_points as proxy for xP)
        optimizer_input = latest[["element", "name", "position", "team"]].copy()
        optimizer_input["xP"] = latest["total_points_roll_4"].fillna(2.0)
        optimizer_input["now_cost"] = latest["value"].fillna(50)

        # Optimize
        result = optimize_team(optimizer_input)

        assert len(result["squad"]) == 15
        assert len(result["xi"]) == 11
        assert result["total_xp"] > 0

        # Position sanity
        xi_pos = result["xi"]["position"].value_counts()
        assert xi_pos["GK"] == 1
        assert 3 <= xi_pos.get("DEF", 0) <= 5

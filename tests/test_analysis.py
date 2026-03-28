import pytest
import pandas as pd
from src.pipeline.analysis import (
    compute_prediction_misses,
    compute_dream_team,
    format_post_match_summary,
)


class TestComputePredictionMisses:
    def test_identifies_overperformer(self):
        picks_df = pd.DataFrame({
            "element": [1, 2, 3],
            "name": ["Saka", "Haaland", "Palmer"],
            "xP": [6.8, 8.5, 4.2],
            "actual_points": [12, 2, 12],
        })
        misses = compute_prediction_misses(picks_df)
        # Palmer: +7.8 (actual - xP), Saka: +5.2, Haaland: -6.5
        names = [m["name"] for m in misses]
        assert "Haaland" in names
        assert "Palmer" in names
        # Sorted by abs(miss) descending
        assert abs(misses[0]["miss"]) >= abs(misses[1]["miss"])

    def test_miss_is_actual_minus_predicted(self):
        picks_df = pd.DataFrame({
            "element": [1],
            "name": ["Haaland"],
            "xP": [8.5],
            "actual_points": [2],
        })
        misses = compute_prediction_misses(picks_df)
        assert misses[0]["miss"] == pytest.approx(2 - 8.5)


class TestComputeDreamTeam:
    def test_selects_highest_scoring_xi(self):
        live_data = pd.DataFrame({
            "element": range(1, 26),
            "name": [f"P{i}" for i in range(1, 26)],
            "position": (["GK"] * 2 + ["DEF"] * 6 + ["MID"] * 8 + ["FWD"] * 9),
            "total_points": [i * 2 for i in range(1, 26)],
            "team": [f"T{i % 8}" for i in range(1, 26)],
        })
        dream = compute_dream_team(live_data)
        assert len(dream) == 11
        # All elements in dream are from the original data
        assert all(e in live_data["element"].values for e in dream["element"])

    def test_dream_team_valid_formation(self):
        live_data = pd.DataFrame({
            "element": range(1, 26),
            "name": [f"P{i}" for i in range(1, 26)],
            "position": (["GK"] * 2 + ["DEF"] * 6 + ["MID"] * 8 + ["FWD"] * 9),
            "total_points": [i * 2 for i in range(1, 26)],
            "team": [f"T{i % 8}" for i in range(1, 26)],
        })
        dream = compute_dream_team(live_data)
        pos_counts = dream["position"].value_counts()
        assert pos_counts.get("GK", 0) == 1
        assert pos_counts.get("DEF", 0) >= 3
        assert pos_counts.get("MID", 0) >= 2
        assert pos_counts.get("FWD", 0) >= 1

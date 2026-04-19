"""Tests that phase_recommend() saves squad_recommend and xi_recommend CSVs.

Culprit if failing: the squad/xi save block added to phase_recommend() in run.py.
"""
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


PREDICTIONS = pd.DataFrame([
    {"element": 1,  "name": "Flekken",    "position": "GK",  "team": "BRE", "now_cost": 45, "xP": 4.1},
    {"element": 2,  "name": "Walker",     "position": "DEF", "team": "BUR", "now_cost": 44, "xP": 7.6},
    {"element": 3,  "name": "Virgil",     "position": "DEF", "team": "LIV", "now_cost": 63, "xP": 6.5},
    {"element": 4,  "name": "Cucurella",  "position": "DEF", "team": "CHE", "now_cost": 60, "xP": 6.3},
    {"element": 5,  "name": "Fernandes",  "position": "MID", "team": "MUN", "now_cost": 103,"xP": 8.5},
    {"element": 6,  "name": "Semenyo",    "position": "MID", "team": "MCI", "now_cost": 82, "xP": 12.0},
    {"element": 7,  "name": "Amad",       "position": "MID", "team": "MUN", "now_cost": 62, "xP": 9.4},
    {"element": 8,  "name": "Hinshelwood","position": "MID", "team": "BHA", "now_cost": 51, "xP": 9.8},
    {"element": 9,  "name": "Gomez",      "position": "MID", "team": "BHA", "now_cost": 49, "xP": 7.1},
    {"element": 10, "name": "Jesus",      "position": "FWD", "team": "ARS", "now_cost": 64, "xP": 7.1},
    {"element": 11, "name": "Mykolenko",  "position": "FWD", "team": "EVE", "now_cost": 49, "xP": 6.6},
    {"element": 12, "name": "Bayindir",   "position": "GK",  "team": "MUN", "now_cost": 47, "xP": 4.2},
    {"element": 13, "name": "Hume",       "position": "DEF", "team": "SUN", "now_cost": 45, "xP": 6.2},
    {"element": 14, "name": "Andersen",   "position": "DEF", "team": "CPL", "now_cost": 45, "xP": 5.0},
    {"element": 15, "name": "Welbeck",    "position": "FWD", "team": "BHA", "now_cost": 61, "xP": 5.7},
])

MOCK_PLAN = {
    "transfers": [{"transfers": [], "hit_cost": 0, "bank_after": 1.5}],
    "projected_xp": 90.0,
    "hit_cost": 0,
    "bank_after": 1.5,
    "squad_after": list(range(1, 16)),
}


def _mock_user_state():
    state = MagicMock()
    state.current_squad = list(range(1, 16))
    state.bank = 15
    state.free_transfers = 1
    return state


def test_squad_recommend_csv_saved(tmp_path):
    """phase_recommend must save squad_recommend.csv with 15 rows."""
    gw32_dir = tmp_path / "2025-26" / "gw32"
    gw32_dir.mkdir(parents=True)
    PREDICTIONS.to_csv(gw32_dir / "predictions.csv", index=False)

    with patch("src.pipeline.run.RESULTS_DIR", tmp_path), \
         patch("src.pipeline.run.CURRENT_SEASON", "2025-26"), \
         patch("src.pipeline.run.load_user_config", return_value={
             "teams": {"default": {"entry_id": 1}},
             "preferences": {"horizon_gws": 1, "max_hit_points": 8, "fdr_sensitivity": 0.15},
         }), \
         patch("src.pipeline.run.fetch_user_team_state", return_value=_mock_user_state()), \
         patch("src.pipeline.run.fetch_fixtures", return_value=[]), \
         patch("src.pipeline.run.recommend_transfers", return_value=MOCK_PLAN), \
         patch("src.pipeline.run.save_recommend_csv"):
        from src.pipeline.run import phase_recommend
        phase_recommend(target_gw=32)

    squad_path = gw32_dir / "squad_recommend.csv"
    assert squad_path.exists(), "squad_recommend.csv not written"
    df = pd.read_csv(squad_path)
    assert len(df) == 15
    assert "name" in df.columns


def test_xi_recommend_csv_saved(tmp_path):
    """phase_recommend must save xi_recommend.csv with 11 rows."""
    gw32_dir = tmp_path / "2025-26" / "gw32"
    gw32_dir.mkdir(parents=True)
    PREDICTIONS.to_csv(gw32_dir / "predictions.csv", index=False)

    with patch("src.pipeline.run.RESULTS_DIR", tmp_path), \
         patch("src.pipeline.run.CURRENT_SEASON", "2025-26"), \
         patch("src.pipeline.run.load_user_config", return_value={
             "teams": {"default": {"entry_id": 1}},
             "preferences": {"horizon_gws": 1, "max_hit_points": 8, "fdr_sensitivity": 0.15},
         }), \
         patch("src.pipeline.run.fetch_user_team_state", return_value=_mock_user_state()), \
         patch("src.pipeline.run.fetch_fixtures", return_value=[]), \
         patch("src.pipeline.run.recommend_transfers", return_value=MOCK_PLAN), \
         patch("src.pipeline.run.save_recommend_csv"):
        from src.pipeline.run import phase_recommend
        phase_recommend(target_gw=32)

    xi_path = gw32_dir / "xi_recommend.csv"
    assert xi_path.exists(), "xi_recommend.csv not written"
    df = pd.read_csv(xi_path)
    assert len(df) == 11

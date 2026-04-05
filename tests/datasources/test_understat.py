import pytest
import pandas as pd
from unittest.mock import patch

from src.pipeline.datasources.understat import (
    fetch_understat_player_gw_stats,
    compute_team_xgc_per_gw,
)


MOCK_PLAYER_DATA = [
    {"player_id": "1", "player": "Salah", "team": "Liverpool",
     "xG": "0.45", "xA": "0.12", "time": "90",
     "date": "2026-01-01", "id": "fixture_1", "h_team": "Arsenal", "a_team": "Liverpool"},
    {"player_id": "2", "player": "Havertz", "team": "Arsenal",
     "xG": "0.31", "xA": "0.05", "time": "85",
     "date": "2026-01-01", "id": "fixture_1", "h_team": "Arsenal", "a_team": "Liverpool"},
]


def test_fetch_returns_dataframe(tmp_path):
    """fetch_understat_player_gw_stats returns a DataFrame with required columns."""
    with patch(
        "src.pipeline.datasources.understat._fetch_player_grouped_stats_async",
        return_value=MOCK_PLAYER_DATA
    ):
        df = fetch_understat_player_gw_stats(season="2025")
    assert isinstance(df, pd.DataFrame)
    assert {"player_id", "team", "xG", "xA", "date"}.issubset(df.columns)


def test_compute_team_xgc_per_gw():
    """compute_team_xgc_per_gw aggregates xG against each team per match."""
    df = pd.DataFrame(MOCK_PLAYER_DATA)
    df["xG"] = df["xG"].astype(float)
    # Arsenal concedes Salah's 0.45; Liverpool concedes Havertz's 0.31
    team_xgc = compute_team_xgc_per_gw(df)
    assert "team" in team_xgc.columns
    assert "fixture_id" in team_xgc.columns
    assert "xGC" in team_xgc.columns
    arsenal_row = team_xgc[team_xgc["team"] == "Arsenal"]
    assert pytest.approx(arsenal_row["xGC"].values[0], abs=0.01) == 0.31


def test_xgc_non_negative():
    """xGC values must be >= 0."""
    df = pd.DataFrame(MOCK_PLAYER_DATA)
    df["xG"] = df["xG"].astype(float)
    team_xgc = compute_team_xgc_per_gw(df)
    assert (team_xgc["xGC"] >= 0).all()

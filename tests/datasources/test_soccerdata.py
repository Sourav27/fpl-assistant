import pytest
import pandas as pd
from unittest.mock import patch
from src.pipeline.datasources.soccerdata_client import (
    fetch_fotmob_player_minutes,
    cross_validate_with_fpl,
    FotMobReliabilityResult,
)


MOCK_FOTMOB_ROWS = [
    {"player_name": "Salah", "team": "Liverpool",
     "competition": "UEFA Champions League", "date": "2026-03-18",
     "minutes": 90},
    {"player_name": "Havertz", "team": "Arsenal",
     "competition": "Premier League", "date": "2026-03-15",
     "minutes": 85},
]

MOCK_FPL_MINUTES = pd.DataFrame([
    {"web_name": "Havertz", "team": "Arsenal", "date": "2026-03-15", "minutes": 85}
])


def test_fetch_returns_dataframe():
    with patch(
        "src.pipeline.datasources.soccerdata_client._fetch_fotmob_raw",
        return_value=pd.DataFrame(MOCK_FOTMOB_ROWS)
    ):
        df = fetch_fotmob_player_minutes(competitions=["UEFA Champions League"])
    assert isinstance(df, pd.DataFrame)
    assert "minutes" in df.columns
    assert all(df["competition"] == "UEFA Champions League")


def test_fetch_filters_by_competition():
    with patch(
        "src.pipeline.datasources.soccerdata_client._fetch_fotmob_raw",
        return_value=pd.DataFrame(MOCK_FOTMOB_ROWS)
    ):
        df = fetch_fotmob_player_minutes(competitions=["Premier League"])
    assert len(df) == 1
    assert df.iloc[0]["player_name"] == "Havertz"


def test_cross_validate_high_correlation():
    fotmob_pl = pd.DataFrame([
        {"player_name": "Havertz", "team": "Arsenal", "date": "2026-03-15", "minutes": 85},
        {"player_name": "Saka", "team": "Arsenal", "date": "2026-03-15", "minutes": 72}
    ])
    fpl_minutes = pd.DataFrame([
        {"web_name": "Havertz", "team": "Arsenal", "date": "2026-03-15", "minutes": 85},
        {"web_name": "Saka", "team": "Arsenal", "date": "2026-03-15", "minutes": 72}
    ])
    result = cross_validate_with_fpl(fotmob_pl, fpl_minutes)
    assert isinstance(result, FotMobReliabilityResult)
    assert result.mae < 5.0
    assert result.correlation > 0.90


def test_cross_validate_result_fields():
    fotmob_pl = pd.DataFrame([
        {"player_name": "Havertz", "team": "Arsenal", "date": "2026-03-15", "minutes": 85},
        {"player_name": "Saka", "team": "Arsenal", "date": "2026-03-15", "minutes": 72}
    ])
    fpl_minutes = pd.DataFrame([
        {"web_name": "Havertz", "team": "Arsenal", "date": "2026-03-15", "minutes": 85},
        {"web_name": "Saka", "team": "Arsenal", "date": "2026-03-15", "minutes": 72}
    ])
    result = cross_validate_with_fpl(fotmob_pl, fpl_minutes)
    assert hasattr(result, "mae")
    assert hasattr(result, "correlation")
    assert hasattr(result, "n_matched")
    assert hasattr(result, "reliable")

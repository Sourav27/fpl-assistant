"""Tests for the soccerdata.Understat-based understat client.

The module must:
1. Use soccerdata.Understat (synchronous) — NOT understatapi async
2. Return only xg_chain and xg_buildup columns
3. Join dates to GW numbers via a FPL fixtures map
4. Accept season format "2425" (for 2024-25), "2324" (for 2023-24)
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.pipeline.datasources.understat import (
    fetch_understat_xg_chain,
    build_date_gw_map,
    SEASON_FORMAT_EXAMPLES,
)


# Minimal MultiIndex DataFrame matching soccerdata.Understat output shape
def _mock_sd_df():
    idx = pd.MultiIndex.from_tuples(
        [
            ("ENG-Premier League", "2425", "2024-08-16 Arsenal-Wolves", "Arsenal", "Saka"),
            ("ENG-Premier League", "2425", "2024-08-16 Arsenal-Wolves", "Wolves", "Cunha"),
            ("ENG-Premier League", "2425", "2024-08-24 Arsenal-Brighton", "Arsenal", "Saka"),
        ],
        names=["league", "season", "game", "team", "player"],
    )
    return pd.DataFrame(
        {
            "xg": [0.45, 0.12, 0.55],
            "xg_chain": [0.60, 0.20, 0.70],
            "xg_buildup": [0.10, 0.05, 0.15],
            "xa": [0.10, 0.05, 0.12],
            "goals": [1, 0, 1],
            "assists": [0, 0, 1],
        },
        index=idx,
    )


MOCK_FIXTURES = [
    {"event": 1, "kickoff_time": "2024-08-16T19:30:00Z"},
    {"event": 2, "kickoff_time": "2024-08-24T15:00:00Z"},
]


def test_fetch_returns_only_xg_chain_and_xg_buildup():
    """Output must contain only xg_chain, xg_buildup (+ join keys)."""
    mock_sd = MagicMock()
    mock_sd.read_player_match_stats.return_value = _mock_sd_df()

    with patch("src.pipeline.datasources.understat._make_understat_reader", return_value=mock_sd):
        with patch("src.pipeline.datasources.understat._fetch_fixtures_for_season",
                   return_value=MOCK_FIXTURES):
            df = fetch_understat_xg_chain(season="2425")

    assert "xg_chain" in df.columns
    assert "xg_buildup" in df.columns
    # Must NOT contain overlapping FPL columns
    for col in ("xg", "xa", "goals", "assists", "shots", "key_passes",
                "yellow_cards", "red_cards"):
        assert col not in df.columns, f"Column '{col}' should be dropped (FPL overlap)"


def test_fetch_adds_gw_column():
    """Output must include a 'gw' column derived from match date."""
    mock_sd = MagicMock()
    mock_sd.read_player_match_stats.return_value = _mock_sd_df()

    with patch("src.pipeline.datasources.understat._make_understat_reader", return_value=mock_sd):
        with patch("src.pipeline.datasources.understat._fetch_fixtures_for_season",
                   return_value=MOCK_FIXTURES):
            df = fetch_understat_xg_chain(season="2425")

    assert "gw" in df.columns
    assert df[df["player"] == "Saka"]["gw"].iloc[0] == 1


def test_fetch_adds_player_and_team_columns():
    """Output must include player name and team from the MultiIndex."""
    mock_sd = MagicMock()
    mock_sd.read_player_match_stats.return_value = _mock_sd_df()

    with patch("src.pipeline.datasources.understat._make_understat_reader", return_value=mock_sd):
        with patch("src.pipeline.datasources.understat._fetch_fixtures_for_season",
                   return_value=MOCK_FIXTURES):
            df = fetch_understat_xg_chain(season="2425")

    assert "player" in df.columns
    assert "team" in df.columns


def test_build_date_gw_map_basic():
    """build_date_gw_map maps kickoff date strings to GW numbers."""
    gw_map = build_date_gw_map(MOCK_FIXTURES)
    assert gw_map["2024-08-16"] == 1
    assert gw_map["2024-08-24"] == 2


def test_build_date_gw_map_dgw_takes_lower_gw():
    """When two fixtures on same date have different GWs, take the lower GW."""
    fixtures = [
        {"event": 19, "kickoff_time": "2025-01-14T19:30:00Z"},
        {"event": 20, "kickoff_time": "2025-01-14T20:00:00Z"},
    ]
    gw_map = build_date_gw_map(fixtures)
    assert gw_map["2025-01-14"] == 19


def test_season_format_examples_exported():
    """SEASON_FORMAT_EXAMPLES must document the format convention."""
    assert isinstance(SEASON_FORMAT_EXAMPLES, dict)
    assert "2324" in SEASON_FORMAT_EXAMPLES
    assert "2425" in SEASON_FORMAT_EXAMPLES


def test_historical_season_logs_warning(caplog):
    """fetch_understat_xg_chain with a historical season must warn about GW mapping."""
    import logging
    mock_sd = MagicMock()
    mock_sd.read_player_match_stats.return_value = _mock_sd_df()

    with patch("src.pipeline.datasources.understat._make_understat_reader", return_value=mock_sd):
        with patch("src.pipeline.datasources.understat._fetch_fixtures_for_season",
                   return_value=MOCK_FIXTURES):
            with patch("src.pipeline.datasources.understat._current_understat_season",
                       return_value="2526"):  # current is 2526, so "2122" is historical
                with caplog.at_level(logging.WARNING, logger="src.pipeline.datasources.understat"):
                    fetch_understat_xg_chain(season="2122")

    assert any("historical" in r.message.lower() or "GW mapping" in r.message
               for r in caplog.records), "Expected warning about historical season GW mapping"

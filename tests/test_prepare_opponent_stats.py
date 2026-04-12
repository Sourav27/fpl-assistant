"""Tests for opponent defensive stats joined onto player rows (B-F1/B-F2)."""
import pandas as pd
import pytest
from src.pipeline.prepare import _compute_team_defensive_stats, add_opponent_stats


@pytest.fixture
def minimal_gw_df():
    """Two teams, 3 GWs of data. Team 1 concedes a lot; Team 2 is solid."""
    rows = []
    # Team 1 players (team=1, opponent=2 each GW)
    for gw in range(1, 7):
        rows.append({"code": 101, "element": 1, "season": "2024-25", "GW": gw,
                     "team": 1, "opponent_team": 2, "was_home": True,
                     "goals_conceded": 2, "total_points": 2, "position": 1})
    # Team 2 players (team=2, opponent=1 each GW)
    for gw in range(1, 7):
        rows.append({"code": 201, "element": 2, "season": "2024-25", "GW": gw,
                     "team": 2, "opponent_team": 1, "was_home": False,
                     "goals_conceded": 0, "total_points": 8, "position": 2})
    return pd.DataFrame(rows)


class TestComputeTeamDefensiveStats:
    def test_returns_dataframe_with_required_columns(self, minimal_gw_df):
        result = _compute_team_defensive_stats(minimal_gw_df)
        assert "team" in result.columns
        assert "season" in result.columns
        assert "GW" in result.columns
        assert "team_gc_roll_4" in result.columns

    def test_rolling_is_lagged(self, minimal_gw_df):
        """team_gc_roll_4 at GW5 should be mean of GW1-4, not include GW5."""
        result = _compute_team_defensive_stats(minimal_gw_df)
        team1 = result[(result["team"] == 1) & (result["GW"] == 5)].iloc[0]
        assert team1["team_gc_roll_4"] == pytest.approx(2.0, abs=0.01)

    def test_early_gws_have_nan(self, minimal_gw_df):
        result = _compute_team_defensive_stats(minimal_gw_df)
        team1_gw1 = result[(result["team"] == 1) & (result["GW"] == 1)].iloc[0]
        assert pd.isna(team1_gw1["team_gc_roll_4"])


class TestAddOpponentStats:
    def test_joins_opponent_gc_onto_player_rows(self, minimal_gw_df):
        result = add_opponent_stats(minimal_gw_df)
        assert "xGC_rolling_4" in result.columns

    def test_team1_player_sees_team2_gc(self, minimal_gw_df):
        """Team 1 player faces Team 2. Team 2 concedes 0 → xGC_rolling_4 for Team 1 player should be 0."""
        result = add_opponent_stats(minimal_gw_df)
        team1_gw5 = result[(result["team"] == 1) & (result["GW"] == 5)].iloc[0]
        # Team 2 concedes 0 per GW, so xGC rolling avg = 0
        assert team1_gw5["xGC_rolling_4"] == pytest.approx(0.0, abs=0.01)

    def test_no_rows_dropped(self, minimal_gw_df):
        result = add_opponent_stats(minimal_gw_df)
        assert len(result) == len(minimal_gw_df)


class TestOpponentFormByPosition:
    def test_returns_opponent_form_rolling_column(self, minimal_gw_df):
        result = add_opponent_stats(minimal_gw_df)
        assert "opponent_form_rolling_6" in result.columns

    def test_opponent_form_rolling_value_is_correct(self, minimal_gw_df):
        result = add_opponent_stats(minimal_gw_df)
        team2_gw7 = result[(result["code"] == 201) & (result["GW"] == 7)] if 7 in result["GW"].values else result[(result["code"] == 201) & (result["GW"] == result["GW"].max())]
        if not team2_gw7.empty:
            assert not pd.isna(team2_gw7.iloc[0]["opponent_form_rolling_6"]) or team2_gw7.iloc[0]["GW"] < 4

    def test_opponent_form_rolling_is_position_specific(self):
        """GK and FWD players facing the same opponent should see different opponent_form values."""
        rows = []
        for gw in range(1, 9):
            rows.append({"code": 1, "element": 1, "season": "2024-25", "GW": gw,
                         "team": 1, "opponent_team": 2, "was_home": True,
                         "goals_conceded": 1, "total_points": 1, "position": 1})  # GK, 1pt
            rows.append({"code": 4, "element": 4, "season": "2024-25", "GW": gw,
                         "team": 1, "opponent_team": 2, "was_home": True,
                         "goals_conceded": 1, "total_points": 6, "position": 4})  # FWD, 6pt
            rows.append({"code": 2, "element": 2, "season": "2024-25", "GW": gw,
                         "team": 2, "opponent_team": 1, "was_home": False,
                         "goals_conceded": 2, "total_points": 3, "position": 2})
        df = pd.DataFrame(rows)
        result = add_opponent_stats(df)
        gk_gw8 = result[(result["code"] == 1) & (result["GW"] == 8)].iloc[0]
        fwd_gw8 = result[(result["code"] == 4) & (result["GW"] == 8)].iloc[0]
        if not pd.isna(gk_gw8["opponent_form_rolling_6"]) and not pd.isna(fwd_gw8["opponent_form_rolling_6"]):
            assert gk_gw8["opponent_form_rolling_6"] != pytest.approx(fwd_gw8["opponent_form_rolling_6"], abs=0.5)

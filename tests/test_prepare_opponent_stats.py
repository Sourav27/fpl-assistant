"""Tests for opponent defensive stats joined onto player rows (B-F1/B-F2)."""
import pandas as pd
import pytest
from src.pipeline.prepare import _compute_team_defensive_stats, add_opponent_stats, add_opponent_xg_stats


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



class TestAddOpponentXgStats:
    @pytest.fixture
    def xg_df(self):
        """Two teams, 6 GWs, 3 players each side. Team 1 vs Team 2."""
        rows = []
        for gw in range(1, 7):
            for i in range(3):
                rows.append({
                    "code": i, "team": 1, "team_id": 1,
                    "opponent_team": 2, "season": "2023-24",
                    "GW": gw, "expected_goals": 0.5,
                })
            for i in range(3, 6):
                rows.append({
                    "code": i, "team": 2, "team_id": 2,
                    "opponent_team": 1, "season": "2023-24",
                    "GW": gw, "expected_goals": 0.3,
                })
        return pd.DataFrame(rows)

    def test_column_added(self, xg_df):
        result = add_opponent_xg_stats(xg_df)
        assert "opponent_xg_for_roll_4" in result.columns

    def test_no_rows_dropped(self, xg_df):
        result = add_opponent_xg_stats(xg_df)
        assert len(result) == len(xg_df)

    def test_all_team1_players_same_gw_have_same_value(self, xg_df):
        """All team-1 players in the same GW face the same opponent xG-for."""
        result = add_opponent_xg_stats(xg_df)
        for gw in range(3, 7):
            gw_t1 = result[(result["GW"] == gw) & (result["team"] == 1)]
            vals = gw_t1["opponent_xg_for_roll_4"].dropna()
            if len(vals) > 1:
                assert vals.nunique() == 1, f"GW {gw}: team-1 players have different opponent_xg_for_roll_4"

    def test_rolling_is_lagged(self, xg_df):
        """Opponent xg-for at GW5 should use GW1-4 data (lag-1 rolling)."""
        result = add_opponent_xg_stats(xg_df)
        # Team-1 players face team-2 whose xG-for sum per GW = 3 * 0.3 = 0.9
        # Rolling mean of 4 GWs = 0.9
        team1_gw5 = result[(result["team"] == 1) & (result["GW"] == 5)].iloc[0]
        assert team1_gw5["opponent_xg_for_roll_4"] == pytest.approx(0.9, abs=0.01)

    def test_no_expected_goals_column_returns_unchanged(self, xg_df):
        df = xg_df.drop(columns=["expected_goals"])
        result = add_opponent_xg_stats(df)
        assert "opponent_xg_for_roll_4" not in result.columns
        assert len(result) == len(df)

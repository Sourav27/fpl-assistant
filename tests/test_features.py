# tests/test_features.py
import pandas as pd
import numpy as np
import pytest
from src.pipeline.features import (
    add_rolling_features,
    add_momentum_features,
    add_form_features,
    add_saves_rolling,
    add_penalty_taker,
    engineer_features,
)


@pytest.fixture
def player_history():
    """10-GW history for one player."""
    return pd.DataFrame({
        "name": ["Saka"] * 10,
        "element": [3] * 10,
        "season": ["2025-26"] * 10,
        "GW": list(range(1, 11)),
        "total_points": [8, 2, 12, 6, 10, 3, 7, 15, 4, 9],
        "minutes": [90, 90, 90, 75, 90, 45, 90, 90, 60, 90],
        "ict_index": [12.0, 4.0, 15.0, 8.0, 13.0, 3.0, 10.0, 18.0, 5.0, 11.0],
        "bps": [35, 12, 42, 22, 38, 10, 30, 45, 15, 33],
        "goals_scored": [1, 0, 2, 1, 1, 0, 1, 2, 0, 1],
        "assists": [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        "clean_sheets": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        "transfers_in": [50000, 30000, 80000, 20000, 60000, 10000, 40000, 90000, 15000, 55000],
        "transfers_out": [10000, 20000, 5000, 30000, 8000, 25000, 12000, 3000, 35000, 9000],
        "value": [105] * 10,
        "influence": [40.0, 15.0, 55.0, 30.0, 45.0, 10.0, 35.0, 60.0, 18.0, 42.0],
        "creativity": [35.0, 10.0, 40.0, 20.0, 38.0, 8.0, 28.0, 45.0, 12.0, 36.0],
        "threat": [50.0, 20.0, 60.0, 35.0, 52.0, 15.0, 42.0, 65.0, 22.0, 48.0],
    })


class TestRollingFeatures:
    def test_adds_rolling_avg_columns(self, player_history):
        result = add_rolling_features(player_history, windows=[4])
        assert "total_points_roll_4" in result.columns
        assert "minutes_roll_4" in result.columns
        assert "ict_index_roll_4" in result.columns

    def test_rolling_values_are_lagged(self, player_history):
        result = add_rolling_features(player_history, windows=[4])
        # Row at GW5 should average GW1-4, not include GW5
        gw5 = result[result["GW"] == 5].iloc[0]
        expected = (8 + 2 + 12 + 6) / 4  # 7.0
        assert gw5["total_points_roll_4"] == pytest.approx(expected, abs=0.1)

    def test_early_gws_have_nan(self, player_history):
        result = add_rolling_features(player_history, windows=[4])
        assert pd.isna(result[result["GW"] == 1].iloc[0]["total_points_roll_4"])


class TestMomentumFeatures:
    def test_adds_momentum_column(self, player_history):
        df = add_rolling_features(player_history, windows=[4, 8])
        result = add_momentum_features(df)
        assert "total_points_momentum" in result.columns


class TestFormFeatures:
    def test_adds_transfers_net(self, player_history):
        result = add_form_features(player_history)
        assert "transfers_net" in result.columns
        assert result["transfers_net"].iloc[0] == 40000  # 50000 - 10000


class TestRollingFeaturesWithCode:
    def test_uses_code_when_available_to_separate_recycled_elements(self):
        """Two players sharing element=1 in different seasons must not pollute each other's rolling avg.

        FPL recycles element IDs across seasons.  If we group by element, player B
        will incorrectly inherit player A's history.  Grouping by code fixes this.
        """
        # Player A: element=1 in 2022-23, code=111
        player_a = pd.DataFrame({
            "element": [1] * 5, "code": [111] * 5, "season": ["2022-23"] * 5,
            "GW": [1, 2, 3, 4, 5],
            "total_points": [10, 10, 10, 10, 10],
            "minutes": [90] * 5, "ict_index": [5.0] * 5, "bps": [20] * 5,
            "goals_scored": [1] * 5, "assists": [0] * 5, "clean_sheets": [0] * 5,
            "influence": [20.0] * 5, "creativity": [10.0] * 5, "threat": [30.0] * 5,
        })
        # Player B: element=1 in 2024-25, code=222  (same element, different player)
        player_b = pd.DataFrame({
            "element": [1] * 5, "code": [222] * 5, "season": ["2024-25"] * 5,
            "GW": [1, 2, 3, 4, 5],
            "total_points": [2, 2, 2, 2, 2],
            "minutes": [90] * 5, "ict_index": [5.0] * 5, "bps": [20] * 5,
            "goals_scored": [0] * 5, "assists": [0] * 5, "clean_sheets": [0] * 5,
            "influence": [20.0] * 5, "creativity": [10.0] * 5, "threat": [30.0] * 5,
        })
        df = pd.concat([player_a, player_b], ignore_index=True)
        result = add_rolling_features(df, windows=[4])

        # Player B GW5 roll_4 should reflect ONLY player B's history (avg=2.0),
        # not a mix with player A's 10-point history.
        # At GW1, player B has zero prior GW history → roll_4 must be NaN.
        # With wrong element-based grouping, player A's 5 GWs bleed in and give 10.0.
        b_gw1 = result[(result["code"] == 222) & (result["GW"] == 1)].iloc[0]
        assert pd.isna(b_gw1["total_points_roll_4"]), (
            "element ID recycling: player B at GW1 should have NaN roll_4 "
            "(no prior history), but got a value — player A's history leaked in"
        )


class TestEngineerFeatures:
    def test_full_pipeline(self, player_history):
        result = engineer_features(player_history)
        assert len(result) <= len(player_history)
        assert "total_points_roll_4" in result.columns
        assert "transfers_net" in result.columns
        # Should drop rows where rolling features are NaN
        assert not result["total_points_roll_4"].isna().any()


class TestFixtureFeatures:
    """Tests for add_fixture_features() — B-F3."""

    @pytest.fixture
    def fixture_df(self):
        """Single player, 3 GWs: normal, BGW (no fixture), DGW (2 fixtures)."""
        return pd.DataFrame({
            "code": [1, 1, 1, 1],
            "season": ["2024-25"] * 4,
            "GW": [1, 2, 3, 3],             # GW3 has 2 rows (DGW)
            "was_home": [True, False, True, False],
            "kickoff_time": [
                "2024-09-14T15:00:00Z",
                "2024-09-21T15:00:00Z",
                "2024-09-28T12:30:00Z",
                "2024-10-01T19:45:00Z",     # 3 days after fixture 1
            ],
            "total_points": [6, 2, 8, 5],
        })

    def test_is_home_added(self, fixture_df):
        from src.pipeline.features import add_fixture_features
        result = add_fixture_features(fixture_df)
        assert "is_home" in result.columns
        assert result[result["GW"] == 1].iloc[0]["is_home"] == 1

    def test_fixture_count_normal_gw(self, fixture_df):
        from src.pipeline.features import add_fixture_features
        result = add_fixture_features(fixture_df)
        gw1_rows = result[result["GW"] == 1]
        assert gw1_rows.iloc[0]["fixture_count"] == 1

    def test_fixture_count_dgw(self, fixture_df):
        from src.pipeline.features import add_fixture_features
        result = add_fixture_features(fixture_df)
        dgw_rows = result[result["GW"] == 3]
        assert all(dgw_rows["fixture_count"] == 2)

    def test_rest_days_computed_for_fixture_2(self, fixture_df):
        from src.pipeline.features import add_fixture_features
        result = add_fixture_features(fixture_df)
        dgw_rows = result[result["GW"] == 3].sort_values("is_fixture_2")
        fixture_2 = dgw_rows[dgw_rows["is_fixture_2"] == 1].iloc[0]
        assert fixture_2["rest_days"] == pytest.approx(3.0, abs=0.5)

    def test_rest_days_zero_for_fixture_1(self, fixture_df):
        from src.pipeline.features import add_fixture_features
        result = add_fixture_features(fixture_df)
        dgw_rows = result[result["GW"] == 3]
        fixture_1 = dgw_rows[dgw_rows["is_fixture_2"] == 0].iloc[0]
        assert fixture_1["rest_days"] == 0.0


class TestSavesRolling:
    def test_saves_roll_4_column_added(self, player_history):
        df = player_history.copy()
        df["saves"] = [3, 4, 2, 5, 1, 3, 0, 2, 1, 2]
        result = add_saves_rolling(df)
        assert "saves_roll_4" in result.columns

    def test_saves_roll_4_is_lagged(self, player_history):
        df = player_history.copy()
        df["saves"] = [3, 4, 2, 5, 1, 3, 0, 2, 1, 2]
        result = add_saves_rolling(df)
        # GW5 should see saves from GW1-4 (min_periods=4, so first valid value is at GW5)
        gw5 = result[result["GW"] == 5].iloc[0]
        expected = (3 + 4 + 2 + 5) / 4  # 3.5
        assert gw5["saves_roll_4"] == pytest.approx(expected, abs=0.01)

    def test_saves_roll_4_early_gws_have_nan(self, player_history):
        """saves_roll_4 should be NaN when fewer than 4 prior GWs exist."""
        df = player_history.copy()
        df["saves"] = [1.0] * len(df)
        result = add_saves_rolling(df)
        # First 4 rows: GW1 has 0 prior, GW2 has 1 prior, GW3 has 2 prior, GW4 has 3 prior
        # All should be NaN with min_periods=4
        assert result["saves_roll_4"].iloc[:4].isna().all(), \
            "saves_roll_4 should be NaN for first 4 rows (min_periods=4)"
        # Row 5+ (GW5 onward) should have a value
        assert result["saves_roll_4"].iloc[4:].notna().any()

    def test_saves_roll_4_no_saves_column(self, player_history):
        """If saves column is absent, dataframe is returned unchanged."""
        result = add_saves_rolling(player_history)
        assert "saves_roll_4" not in result.columns

    def test_saves_roll_4_via_engineer_features(self, player_history):
        df = player_history.copy()
        df["saves"] = [3, 4, 2, 5, 1, 3, 0, 2, 1, 2]
        result = engineer_features(df)
        assert "saves_roll_4" in result.columns
        assert result["saves_roll_4"].notna().any()


class TestPenaltyTaker:
    def test_penalty_taker_column_added(self, player_history):
        df = player_history.copy()
        df["penalties_order"] = [1, 2, None, 1, 3, 2, 1, None, 2, 3]
        result = add_penalty_taker(df)
        assert "penalty_taker" in result.columns

    def test_penalty_taker_binary_values(self, player_history):
        df = player_history.copy()
        df["penalties_order"] = [1, 2, None, 1, 3, 2, 1, None, 2, 3]
        result = add_penalty_taker(df)
        first_takers = result[df["penalties_order"] == 1]["penalty_taker"]
        assert first_takers.eq(1).all()
        non_takers = result[df["penalties_order"].fillna(0) != 1]["penalty_taker"]
        assert non_takers.eq(0).all()

    def test_penalty_taker_zero_when_column_absent(self, player_history):
        result = add_penalty_taker(player_history)
        assert "penalty_taker" in result.columns
        assert result["penalty_taker"].eq(0).all()

    def test_penalty_taker_via_engineer_features(self, player_history):
        df = player_history.copy()
        df["penalties_order"] = [1, 2, None, 1, 3, 2, 1, None, 2, 3]
        result = engineer_features(df)
        assert "penalty_taker" in result.columns

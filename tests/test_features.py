# tests/test_features.py
import pandas as pd
import numpy as np
import pytest
from src.pipeline.features import (
    add_rolling_features,
    add_momentum_features,
    add_form_features,
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


class TestEngineerFeatures:
    def test_full_pipeline(self, player_history):
        result = engineer_features(player_history)
        assert len(result) <= len(player_history)
        assert "total_points_roll_4" in result.columns
        assert "transfers_net" in result.columns
        # Should drop rows where rolling features are NaN
        assert not result["total_points_roll_4"].isna().any()

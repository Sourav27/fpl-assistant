# tests/test_predict_position.py
"""Tests for per-position model routing and DGW aggregation (B-F4, B-F3 prediction side)."""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def players_multi_position():
    """One player per position for routing tests."""
    base_features = {col: [0.0] for col in [
        "total_points_roll_4", "total_points_roll_8",
        "minutes_roll_4", "minutes_roll_8",
        "ict_index_roll_4", "ict_index_roll_8",
        "bps_roll_4", "bps_roll_8",
        "goals_scored_roll_4", "assists_roll_4",
        "clean_sheets_roll_4",
        "influence_roll_4", "creativity_roll_4", "threat_roll_4",
        "total_points_momentum", "minutes_momentum", "ict_index_momentum",
        "transfers_net",
        # new fixture features
        "xGC_rolling_4", "opponent_form_rolling_6",
        "is_home", "fixture_count", "rest_days", "is_fixture_2",
    ]}
    rows = []
    for pos, el in [("GK", 1), ("DEF", 2), ("MID", 3), ("FWD", 4)]:
        row = {"element": el, "code": el * 100, "name": f"Player{pos}",
               "position": pos, "team": 1, "now_cost": 55}
        row.update({k: v[0] for k, v in base_features.items()})
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def dgw_players():
    """One MID player with 2 fixture rows for DGW (GW rows pre-expanded)."""
    feat_vals = {col: 0.0 for col in [
        "total_points_roll_4", "total_points_roll_8",
        "minutes_roll_4", "minutes_roll_8",
        "ict_index_roll_4", "ict_index_roll_8",
        "bps_roll_4", "bps_roll_8",
        "goals_scored_roll_4", "assists_roll_4",
        "clean_sheets_roll_4",
        "influence_roll_4", "creativity_roll_4", "threat_roll_4",
        "total_points_momentum", "minutes_momentum", "ict_index_momentum",
        "transfers_net",
        "xGC_rolling_4", "opponent_form_rolling_6",
        "is_home", "rest_days",
    ]}
    row1 = {"element": 10, "code": 1000, "name": "Saka", "position": "MID",
            "team": 3, "now_cost": 105, "fixture_count": 2, "is_fixture_2": 0}
    row2 = {"element": 10, "code": 1000, "name": "Saka", "position": "MID",
            "team": 3, "now_cost": 105, "fixture_count": 2, "is_fixture_2": 1,
            "rest_days": 3.0}
    row1.update(feat_vals)
    row2.update(feat_vals)
    return pd.DataFrame([row1, row2])


class TestPositionRouting:
    def test_routes_gk_to_gk_model(self, players_multi_position, tmp_path):
        """Each position's player should be predicted by its position model."""
        from src.pipeline.predict import predict_next_gw_per_position

        mock_model = MagicMock()
        mock_model.predict.return_value = [5.0]

        mock_models = {pos: mock_model for pos in ["GK", "DEF", "MID", "FWD"]}
        result = predict_next_gw_per_position(players_multi_position, models=mock_models)

        assert len(result) == 4
        assert set(result["position"]) == {"GK", "DEF", "MID", "FWD"}

    def test_fallback_when_model_missing_uses_ep_next(self, players_multi_position):
        """If a position model is None, xP falls back to ep_next when ep_next_map provided."""
        from src.pipeline.predict import predict_next_gw_per_position

        mock_model = MagicMock()
        mock_model.predict.return_value = [5.0]
        # GK model missing; GK player has element=1
        models = {"GK": None, "DEF": mock_model, "MID": mock_model, "FWD": mock_model}
        ep_next_map = {1: 3.7}  # element 1 is the GK
        result = predict_next_gw_per_position(players_multi_position, models=models, ep_next_map=ep_next_map)
        gk_row = result[result["position"] == "GK"].iloc[0]
        assert gk_row["xP"] == pytest.approx(3.7, abs=0.01)

    def test_fallback_when_model_missing_and_no_ep_next(self, players_multi_position):
        """If a position model is None and no ep_next_map, xP is 0 (safe default)."""
        from src.pipeline.predict import predict_next_gw_per_position

        mock_model = MagicMock()
        mock_model.predict.return_value = [5.0]
        models = {"GK": None, "DEF": mock_model, "MID": mock_model, "FWD": mock_model}
        result = predict_next_gw_per_position(players_multi_position, models=models)
        gk_row = result[result["position"] == "GK"].iloc[0]
        assert gk_row["xP"] == pytest.approx(0.0, abs=0.01)


class TestDGWAggregation:
    def test_dgw_player_xp_is_sum_of_two_fixtures(self, dgw_players):
        """A DGW player with 2 fixture rows should have xP = sum of both predictions."""
        from src.pipeline.predict import predict_next_gw_per_position

        mock_model = MagicMock()
        mock_model.predict.return_value = [4.0]  # each fixture returns 4

        models = {"GK": mock_model, "DEF": mock_model, "MID": mock_model, "FWD": mock_model}
        result = predict_next_gw_per_position(dgw_players, models=models)

        assert len(result) == 1  # one row per player
        assert result.iloc[0]["xP"] == pytest.approx(8.0, abs=0.01)  # 4+4

    def test_single_fixture_not_doubled(self, players_multi_position):
        """fixture_count=1 players must not have their xP doubled."""
        from src.pipeline.predict import predict_next_gw_per_position

        mock_model = MagicMock()
        mock_model.predict.return_value = [3.0]

        # Ensure fixture_count=1 for all
        df = players_multi_position.copy()
        df["fixture_count"] = 1
        df["is_fixture_2"] = 0
        df["rest_days"] = 0.0

        models = {pos: mock_model for pos in ["GK", "DEF", "MID", "FWD"]}
        result = predict_next_gw_per_position(df, models=models)
        for _, row in result.iterrows():
            assert row["xP"] == pytest.approx(3.0, abs=0.01)


@pytest.mark.skip(reason="B-F3-DGW: phase_predict fixture expansion not yet implemented")
def test_phase_predict_expands_dgw_player_to_two_rows():
    pass

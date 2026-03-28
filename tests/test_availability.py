# tests/test_availability.py
import pandas as pd
import pytest
from src.pipeline.availability import filter_availability


@pytest.fixture
def predictions_with_availability():
    """Predictions DataFrame with varying availability statuses."""
    return pd.DataFrame({
        "element": [1, 2, 3, 4, 5, 6, 7],
        "name": ["Available", "Injured", "Doubtful75", "Doubtful50", "Suspended", "DoubtfulNull", "Available100"],
        "position": ["MID"] * 7,
        "team": ["Arsenal"] * 7,
        "xP": [6.0, 5.0, 7.0, 4.0, 3.0, 5.5, 8.0],
        "now_cost": [100] * 7,
    })


@pytest.fixture
def bootstrap_with_availability():
    """Bootstrap data with various availability statuses."""
    return {
        "elements": [
            {"id": 1, "status": "a", "chance_of_playing_next_round": None, "news": ""},
            {"id": 2, "status": "i", "chance_of_playing_next_round": 0, "news": "Knee injury"},
            {"id": 3, "status": "d", "chance_of_playing_next_round": 75, "news": "Hamstring - 75%"},
            {"id": 4, "status": "d", "chance_of_playing_next_round": 50, "news": "Illness - 50%"},
            {"id": 5, "status": "s", "chance_of_playing_next_round": 0, "news": "Suspended"},
            {"id": 6, "status": "d", "chance_of_playing_next_round": None, "news": "Knock"},
            {"id": 7, "status": "a", "chance_of_playing_next_round": 100, "news": ""},
        ],
    }


class TestFilterAvailability:
    def test_excludes_injured_players(self, predictions_with_availability, bootstrap_with_availability):
        result = filter_availability(predictions_with_availability, bootstrap_with_availability)
        assert 2 not in result["element"].values  # Injured

    def test_excludes_suspended_players(self, predictions_with_availability, bootstrap_with_availability):
        result = filter_availability(predictions_with_availability, bootstrap_with_availability)
        assert 5 not in result["element"].values  # Suspended

    def test_keeps_available_players_unchanged(self, predictions_with_availability, bootstrap_with_availability):
        result = filter_availability(predictions_with_availability, bootstrap_with_availability)
        avail = result[result["element"] == 1]
        assert len(avail) == 1
        assert avail.iloc[0]["xP"] == 6.0  # unchanged

    def test_scales_75_percent_chance(self, predictions_with_availability, bootstrap_with_availability):
        result = filter_availability(predictions_with_availability, bootstrap_with_availability)
        d75 = result[result["element"] == 3]
        assert len(d75) == 1
        assert d75.iloc[0]["xP"] == pytest.approx(7.0 * 0.75)

    def test_scales_50_percent_chance(self, predictions_with_availability, bootstrap_with_availability):
        result = filter_availability(predictions_with_availability, bootstrap_with_availability)
        d50 = result[result["element"] == 4]
        assert len(d50) == 1
        assert d50.iloc[0]["xP"] == pytest.approx(4.0 * 0.50)

    def test_doubtful_null_treated_as_50(self, predictions_with_availability, bootstrap_with_availability):
        """Rule 4: status='d' + chance=null → xP * 0.50"""
        result = filter_availability(predictions_with_availability, bootstrap_with_availability)
        d_null = result[result["element"] == 6]
        assert len(d_null) == 1
        assert d_null.iloc[0]["xP"] == pytest.approx(5.5 * 0.50)

    def test_available_100_unchanged(self, predictions_with_availability, bootstrap_with_availability):
        result = filter_availability(predictions_with_availability, bootstrap_with_availability)
        a100 = result[result["element"] == 7]
        assert a100.iloc[0]["xP"] == 8.0

    def test_returns_fewer_rows_than_input(self, predictions_with_availability, bootstrap_with_availability):
        result = filter_availability(predictions_with_availability, bootstrap_with_availability)
        # Should exclude 2 players (injured + suspended)
        assert len(result) == 5

    def test_logs_excluded_players(self, predictions_with_availability, bootstrap_with_availability, caplog):
        import logging
        with caplog.at_level(logging.INFO):
            filter_availability(predictions_with_availability, bootstrap_with_availability)
        # Should log at least 2 exclusions (injured + suspended)
        exclude_logs = [r for r in caplog.records if "Excluded" in r.message or "Scaled" in r.message]
        assert len(exclude_logs) >= 2

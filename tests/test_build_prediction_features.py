# tests/test_build_prediction_features.py
"""TDD tests for build_prediction_features — prediction-time fixture expansion."""
import pandas as pd
import pytest
from src.pipeline.features import build_prediction_features


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def teams():
    return [
        {"id": 1, "name": "Arsenal"},
        {"id": 2, "name": "Man City"},
        {"id": 3, "name": "Brighton"},
        {"id": 4, "name": "Wolves"},  # present in fixtures but not in latest
    ]


@pytest.fixture
def sgw_fixtures():
    """Each team in latest plays exactly once in GW33.
    Arsenal(1) hosts Man City(2). Brighton(3) hosts Wolves(4).
    Wolves is not in latest so only 3 rows expected.
    """
    return [
        {"event": 33, "team_h": 1, "team_a": 2, "kickoff_time": "2026-04-26T14:00:00Z"},
        {"event": 33, "team_h": 3, "team_a": 4, "kickoff_time": "2026-04-27T14:00:00Z"},
        {"event": 34, "team_h": 2, "team_a": 3, "kickoff_time": "2026-05-03T14:00:00Z"},
    ]


@pytest.fixture
def dgw_fixtures():
    """Man City and Brighton play twice in GW33. Arsenal plays once.
    Man City: away Apr 26 (vs Wolves), home Apr 28 (vs Brighton) → is_home=[0,1].
    Brighton: away Apr 27 (vs Arsenal), away Apr 28 (vs Man City) → 2 fixtures.
    Arsenal: home Apr 27 (vs Brighton) → 1 fixture.
    """
    return [
        {"event": 33, "team_h": 4, "team_a": 2, "kickoff_time": "2026-04-26T14:00:00Z"},
        {"event": 33, "team_h": 1, "team_a": 3, "kickoff_time": "2026-04-27T14:00:00Z"},
        {"event": 33, "team_h": 2, "team_a": 3, "kickoff_time": "2026-04-28T19:45:00Z"},
    ]


@pytest.fixture
def latest():
    """One row per player — rolling features already computed."""
    return pd.DataFrame({
        "element": [1, 2, 3],
        "code": [100, 200, 300],
        "name": ["Raya", "Haaland", "Mitoma"],
        "position": ["GK", "FWD", "MID"],
        "team": ["Arsenal", "Man City", "Brighton"],
        "now_cost": [60, 145, 61],
        "total_points_roll_4": [3.0, 5.0, 4.0],
        "minutes_roll_4": [90.0, 88.0, 85.0],
    })


# ---------------------------------------------------------------------------
# Single-gameweek: one row per player
# ---------------------------------------------------------------------------

class TestSGW:
    def test_returns_one_row_per_player(self, latest, sgw_fixtures, teams):
        result = build_prediction_features(latest, sgw_fixtures, 33, teams)
        assert len(result) == len(latest)

    def test_preserves_rolling_features(self, latest, sgw_fixtures, teams):
        result = build_prediction_features(latest, sgw_fixtures, 33, teams)
        raya = result[result["name"] == "Raya"].iloc[0]
        assert raya["total_points_roll_4"] == pytest.approx(3.0)

    def test_fixture_count_is_one(self, latest, sgw_fixtures, teams):
        result = build_prediction_features(latest, sgw_fixtures, 33, teams)
        assert (result["fixture_count"] == 1).all()

    def test_is_fixture_2_is_zero(self, latest, sgw_fixtures, teams):
        result = build_prediction_features(latest, sgw_fixtures, 33, teams)
        assert (result["is_fixture_2"] == 0).all()

    def test_rest_days_is_zero(self, latest, sgw_fixtures, teams):
        result = build_prediction_features(latest, sgw_fixtures, 33, teams)
        assert (result["rest_days"] == 0.0).all()

    def test_is_home_set_correctly(self, latest, sgw_fixtures, teams):
        # Arsenal is team_h → is_home=1; Man City is team_a → is_home=0
        result = build_prediction_features(latest, sgw_fixtures, 33, teams)
        assert result[result["name"] == "Raya"].iloc[0]["is_home"] == 1
        assert result[result["name"] == "Haaland"].iloc[0]["is_home"] == 0
        assert result[result["name"] == "Mitoma"].iloc[0]["is_home"] == 1


# ---------------------------------------------------------------------------
# Double-gameweek: DGW players get 2 rows, SGW players get 1
# ---------------------------------------------------------------------------

class TestDGW:
    def test_dgw_players_get_two_rows(self, latest, dgw_fixtures, teams):
        result = build_prediction_features(latest, dgw_fixtures, 33, teams)
        haaland_rows = result[result["name"] == "Haaland"]
        assert len(haaland_rows) == 2

    def test_sgw_player_gets_one_row(self, latest, dgw_fixtures, teams):
        # Arsenal only plays once in dgw_fixtures GW33
        result = build_prediction_features(latest, dgw_fixtures, 33, teams)
        raya_rows = result[result["name"] == "Raya"]
        assert len(raya_rows) == 1

    def test_fixture_count_is_two_for_dgw(self, latest, dgw_fixtures, teams):
        result = build_prediction_features(latest, dgw_fixtures, 33, teams)
        haaland_rows = result[result["name"] == "Haaland"]
        assert (haaland_rows["fixture_count"] == 2).all()

    def test_is_fixture_2_correct_for_dgw(self, latest, dgw_fixtures, teams):
        result = build_prediction_features(latest, dgw_fixtures, 33, teams)
        haaland_rows = result[result["name"] == "Haaland"].sort_values("is_fixture_2")
        assert list(haaland_rows["is_fixture_2"]) == [0, 1]

    def test_rest_days_zero_for_first_fixture(self, latest, dgw_fixtures, teams):
        result = build_prediction_features(latest, dgw_fixtures, 33, teams)
        first = result[(result["name"] == "Haaland") & (result["is_fixture_2"] == 0)].iloc[0]
        assert first["rest_days"] == pytest.approx(0.0)

    def test_rest_days_positive_for_second_fixture(self, latest, dgw_fixtures, teams):
        # Fixtures on Apr 26 and Apr 28 → ~2 days apart
        result = build_prediction_features(latest, dgw_fixtures, 33, teams)
        second = result[(result["name"] == "Haaland") & (result["is_fixture_2"] == 1)].iloc[0]
        # Apr 26 14:00 → Apr 28 19:45 = ~2.24 days
        assert second["rest_days"] == pytest.approx(2.24, abs=0.1)

    def test_is_home_set_per_fixture(self, latest, dgw_fixtures, teams):
        # Man City: away in fixture 1 (team_a), home in fixture 2 (team_h)
        result = build_prediction_features(latest, dgw_fixtures, 33, teams)
        haaland_rows = result[result["name"] == "Haaland"].sort_values("is_fixture_2")
        assert list(haaland_rows["is_home"]) == [0, 1]

    def test_rolling_features_identical_across_dgw_rows(self, latest, dgw_fixtures, teams):
        result = build_prediction_features(latest, dgw_fixtures, 33, teams)
        haaland_rows = result[result["name"] == "Haaland"]
        assert haaland_rows["total_points_roll_4"].nunique() == 1

    def test_total_rows_correct(self, latest, dgw_fixtures, teams):
        # Arsenal: 1 fixture, Man City: 2, Brighton: 2 → total 5 rows
        result = build_prediction_features(latest, dgw_fixtures, 33, teams)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_player_with_no_fixture_excluded(self, teams):
        """A player whose team has no fixture in target_gw is dropped."""
        latest = pd.DataFrame({
            "element": [1, 2],
            "code": [100, 200],
            "name": ["Raya", "Haaland"],
            "position": ["GK", "FWD"],
            "team": ["Arsenal", "Man City"],
            "now_cost": [60, 145],
            "total_points_roll_4": [3.0, 5.0],
            "minutes_roll_4": [90.0, 88.0],
        })
        # Only Arsenal has a fixture — Man City has a blank GW
        fixtures = [
            {"event": 33, "team_h": 1, "team_a": 3, "kickoff_time": "2026-04-26T14:00:00Z"},
        ]
        result = build_prediction_features(latest, fixtures, 33, teams)
        assert "Haaland" not in result["name"].values
        assert len(result) == 1

    def test_empty_latest_returns_empty(self, sgw_fixtures, teams):
        empty = pd.DataFrame(columns=["element", "code", "name", "position", "team", "now_cost"])
        result = build_prediction_features(empty, sgw_fixtures, 33, teams)
        assert result.empty

    def test_filters_to_correct_gw(self, latest, teams):
        """Fixtures from other GWs must be ignored."""
        fixtures = [
            {"event": 34, "team_h": 1, "team_a": 2, "kickoff_time": "2026-05-03T14:00:00Z"},
        ]
        result = build_prediction_features(latest, fixtures, 33, teams)
        assert result.empty

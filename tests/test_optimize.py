# tests/test_optimize.py
import pandas as pd
import pytest
from src.pipeline.optimize import (
    select_squad,
    select_xi,
    select_captain,
    optimize_team,
)
from src.config import SQUAD_RULES


@pytest.fixture
def player_pool():
    """20-player pool with realistic positions and costs."""
    return pd.DataFrame({
        "element": range(1, 21),
        "name": [
            "GK1", "GK2", "GK3",
            "DEF1", "DEF2", "DEF3", "DEF4", "DEF5", "DEF6",
            "MID1", "MID2", "MID3", "MID4", "MID5", "MID6",
            "FWD1", "FWD2", "FWD3", "FWD4", "FWD5",
        ],
        "position": (
            ["GK"] * 3 + ["DEF"] * 6 + ["MID"] * 6 + ["FWD"] * 5
        ),
        "team": [
            "A", "B", "C",
            "A", "B", "C", "D", "E", "F",
            "A", "B", "C", "D", "E", "F",
            "A", "B", "C", "D", "E",
        ],
        "xP": [
            4.0, 3.5, 3.0,
            5.5, 5.0, 4.8, 4.5, 4.0, 3.5,
            7.0, 6.5, 6.0, 5.5, 5.0, 4.5,
            8.0, 7.5, 6.0, 5.0, 4.0,
        ],
        "now_cost": [
            45, 40, 40,
            60, 55, 55, 50, 48, 45,
            100, 90, 85, 75, 70, 65,
            130, 110, 80, 70, 60,
        ],
    })


class TestSelectSquad:
    def test_returns_15_players(self, player_pool):
        squad = select_squad(player_pool)
        assert len(squad) == 15

    def test_respects_position_constraints(self, player_pool):
        squad = select_squad(player_pool)
        pos_counts = squad["position"].value_counts()
        assert pos_counts.get("GK", 0) == 2
        assert pos_counts.get("DEF", 0) == 5
        assert pos_counts.get("MID", 0) == 5
        assert pos_counts.get("FWD", 0) == 3

    def test_respects_budget(self, player_pool):
        squad = select_squad(player_pool)
        assert squad["now_cost"].sum() <= SQUAD_RULES["budget"]

    def test_max_3_per_team(self, player_pool):
        squad = select_squad(player_pool)
        team_counts = squad["team"].value_counts()
        assert team_counts.max() <= 3

    def test_maximizes_xp(self, player_pool):
        squad = select_squad(player_pool)
        # The optimizer should pick high-xP players
        assert squad["xP"].sum() > 60  # reasonable lower bound


class TestSelectXI:
    def test_returns_11_players(self, player_pool):
        squad = select_squad(player_pool)
        xi = select_xi(squad)
        assert len(xi) == 11

    def test_valid_formation(self, player_pool):
        squad = select_squad(player_pool)
        xi = select_xi(squad)
        pos = xi["position"].value_counts()
        assert pos.get("GK", 0) == 1
        assert 3 <= pos.get("DEF", 0) <= 5
        assert 2 <= pos.get("MID", 0) <= 5
        assert 1 <= pos.get("FWD", 0) <= 3


class TestSelectCaptain:
    def test_captain_has_highest_xp(self, player_pool):
        squad = select_squad(player_pool)
        xi = select_xi(squad)
        captain, vice = select_captain(xi)
        assert captain["xP"] >= vice["xP"]
        assert captain["xP"] == xi["xP"].max()


class TestOptimizeTeam:
    def test_full_pipeline(self, player_pool):
        result = optimize_team(player_pool)
        assert "squad" in result
        assert "xi" in result
        assert "captain" in result
        assert "vice_captain" in result
        assert "total_xp" in result
        assert len(result["squad"]) == 15
        assert len(result["xi"]) == 11

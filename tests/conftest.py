import pytest
import pandas as pd

@pytest.fixture
def sample_bootstrap_json():
    """Minimal FPL API bootstrap-static response with availability fields."""
    return {
        "events": [
            {"id": 29, "deadline_time": "2026-03-07T11:30:00Z", "is_current": False, "is_next": False, "finished": True},
            {"id": 30, "deadline_time": "2026-03-14T11:00:00Z", "is_current": True, "is_next": False, "finished": False},
            {"id": 31, "deadline_time": "2026-03-20T18:30:00Z", "is_current": False, "is_next": True, "finished": False},
        ],
        "elements": [
            {
                "id": 1, "first_name": "David", "second_name": "Raya",
                "web_name": "Raya", "team": 1, "element_type": 1,
                "now_cost": 55, "total_points": 120, "minutes": 2700,
                "ep_this": "4.2", "ep_next": "4.5",
                "status": "a", "chance_of_playing_next_round": None,
                "news": "", "news_added": None,
                "form": "5.0", "selected_by_percent": "25.0",
                "goals_scored": 0, "assists": 0, "clean_sheets": 12,
                "expected_goals": "0.0", "expected_assists": "0.1",
            },
            {
                "id": 2, "first_name": "Gabriel", "second_name": "Magalhaes",
                "web_name": "Gabriel", "team": 1, "element_type": 2,
                "now_cost": 62, "total_points": 140, "minutes": 2600,
                "ep_this": "5.1", "ep_next": "5.3",
                "status": "a", "chance_of_playing_next_round": 100,
                "news": "", "news_added": None,
                "form": "6.0", "selected_by_percent": "30.0",
                "goals_scored": 4, "assists": 1, "clean_sheets": 12,
                "expected_goals": "3.2", "expected_assists": "0.8",
            },
            {
                "id": 3, "first_name": "Bukayo", "second_name": "Saka",
                "web_name": "Saka", "team": 1, "element_type": 3,
                "now_cost": 105, "total_points": 180, "minutes": 2400,
                "ep_this": "6.8", "ep_next": "7.0",
                "status": "d", "chance_of_playing_next_round": 75,
                "news": "Hamstring - 75% chance of playing", "news_added": "2026-03-12T10:00:00Z",
                "form": "7.5", "selected_by_percent": "45.0",
                "goals_scored": 12, "assists": 10, "clean_sheets": 0,
                "expected_goals": "10.5", "expected_assists": "8.2",
            },
            {
                "id": 4, "first_name": "Martin", "second_name": "Odegaard",
                "web_name": "Odegaard", "team": 1, "element_type": 3,
                "now_cost": 82, "total_points": 90, "minutes": 1800,
                "ep_this": "3.5", "ep_next": "3.8",
                "status": "i", "chance_of_playing_next_round": 0,
                "news": "Knee injury - expected back April 2026", "news_added": "2026-02-20T14:00:00Z",
                "form": "2.0", "selected_by_percent": "15.0",
                "goals_scored": 5, "assists": 7, "clean_sheets": 0,
                "expected_goals": "4.0", "expected_assists": "6.0",
            },
        ],
        "teams": [
            {"id": 1, "name": "Arsenal", "short_name": "ARS", "code": 3,
             "strength": 5, "strength_attack_home": 1340, "strength_attack_away": 1390,
             "strength_defence_home": 1260, "strength_defence_away": 1340},
        ],
        "element_types": [
            {"id": 1, "singular_name": "Goalkeeper", "singular_name_short": "GKP", "plural_name_short": "GKP"},
            {"id": 2, "singular_name": "Defender", "singular_name_short": "DEF", "plural_name_short": "DEF"},
            {"id": 3, "singular_name": "Midfielder", "singular_name_short": "MID", "plural_name_short": "MID"},
            {"id": 4, "singular_name": "Forward", "singular_name_short": "FWD", "plural_name_short": "FWD"},
        ],
    }

@pytest.fixture
def sample_player_history_json():
    """Minimal FPL API element-summary response for one player."""
    return {
        "history": [
            {
                "element": 3, "fixture": 1, "opponent_team": 10,
                "total_points": 8, "was_home": True, "kickoff_time": "2025-08-16T14:00:00Z",
                "round": 1, "minutes": 90, "goals_scored": 1, "assists": 1,
                "clean_sheets": 0, "goals_conceded": 1, "bonus": 3, "bps": 35,
                "influence": "40.0", "creativity": "35.0", "threat": "50.0",
                "ict_index": "12.5", "starts": 1, "expected_goals": "0.8",
                "expected_assists": "0.5", "expected_goal_involvements": "1.3",
                "expected_goals_conceded": "1.2",
                "value": 100, "transfers_balance": 40000,
                "transfers_in": 50000, "transfers_out": 10000, "selected": 3000000,
            },
        ],
        "history_past": [],
        "fixtures": [],
    }

@pytest.fixture
def sample_gw_df():
    """Sample gameweek DataFrame matching vaastav merged_gw.csv schema."""
    return pd.DataFrame({
        "name": ["Saka", "Saka", "Saka", "Saka", "Gabriel", "Gabriel"],
        "position": ["MID", "MID", "MID", "MID", "DEF", "DEF"],
        "team": ["Arsenal", "Arsenal", "Arsenal", "Arsenal", "Arsenal", "Arsenal"],
        "xP": [6.5, 5.2, 7.1, 4.8, 4.0, 3.5],
        "element": [3, 3, 3, 3, 2, 2],
        "total_points": [8, 2, 12, 6, 6, 2],
        "minutes": [90, 90, 90, 75, 90, 90],
        "goals_scored": [1, 0, 2, 1, 1, 0],
        "assists": [1, 0, 1, 0, 0, 0],
        "clean_sheets": [0, 1, 0, 0, 1, 0],
        "ict_index": [12.5, 4.2, 15.0, 8.3, 6.0, 3.1],
        "influence": [40.0, 15.0, 55.0, 30.0, 25.0, 10.0],
        "creativity": [35.0, 10.0, 40.0, 20.0, 5.0, 3.0],
        "threat": [50.0, 20.0, 60.0, 35.0, 30.0, 18.0],
        "bps": [35, 12, 42, 22, 28, 15],
        "bonus": [3, 0, 3, 1, 2, 0],
        "value": [105, 105, 106, 106, 60, 61],
        "transfers_in": [50000, 30000, 80000, 20000, 15000, 10000],
        "transfers_out": [10000, 20000, 5000, 30000, 5000, 8000],
        "selected": [3000000, 3100000, 3200000, 3150000, 2000000, 2050000],
        "was_home": [True, False, True, False, True, False],
        "opponent_team": [10, 15, 8, 20, 10, 15],
        "round": [26, 27, 28, 29, 28, 29],
        "GW": [26, 27, 28, 29, 28, 29],
        "season": ["2025-26"] * 6,
    })

@pytest.fixture
def sample_predictions_df():
    """Sample predictions DataFrame for optimizer input."""
    return pd.DataFrame({
        "element": range(1, 16),
        "name": [
            "Raya", "Martinez", "Gabriel", "Saliba", "Alexander-Arnold",
            "Estupinan", "Van Dijk", "Saka", "Palmer", "Salah",
            "Mbeumo", "Gordon", "Haaland", "Watkins", "Isak",
        ],
        "position": [
            "GK", "GK", "DEF", "DEF", "DEF",
            "DEF", "DEF", "MID", "MID", "MID",
            "MID", "MID", "FWD", "FWD", "FWD",
        ],
        "team": [
            "Arsenal", "Aston Villa", "Arsenal", "Arsenal", "Liverpool",
            "Brighton", "Liverpool", "Arsenal", "Chelsea", "Liverpool",
            "Brentford", "Newcastle", "Man City", "Aston Villa", "Newcastle",
        ],
        "xP": [4.2, 3.8, 5.1, 4.8, 6.2, 3.5, 4.9, 6.8, 7.0, 8.5, 4.5, 5.0, 7.5, 5.5, 6.0],
        "now_cost": [55, 48, 62, 58, 72, 50, 65, 105, 100, 130, 78, 72, 140, 82, 88],
    })

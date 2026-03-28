from src.config import (
    DATA_DIR, VAASTAV_DIR, RESULTS_DIR, MODELS_DIR,
    FPL_API_BASE, SEASONS, CURRENT_SEASON,
    SQUAD_RULES, ACTIVE_MODEL,
    AVAILABILITY_HARD_EXCLUDE_STATUS, AVAILABILITY_HARD_EXCLUDE_CHANCE,
)

def test_data_dir_is_relative():
    assert not str(DATA_DIR).startswith("C:")
    assert not str(DATA_DIR).startswith("D:")
    assert DATA_DIR.name == "data"

def test_fpl_api_base_url():
    assert FPL_API_BASE == "https://fantasy.premierleague.com/api"

def test_current_season_format():
    assert len(CURRENT_SEASON) == 7  # "2025-26"
    assert "-" in CURRENT_SEASON

def test_squad_rules():
    assert SQUAD_RULES["squad_size"] == 15
    assert SQUAD_RULES["budget"] == 1000
    assert SQUAD_RULES["max_per_team"] == 3
    assert SQUAD_RULES["positions"] == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

def test_active_model_path():
    assert ACTIVE_MODEL.name == "rf_model.sav"
    assert ACTIVE_MODEL.parent.name == "models"

def test_availability_constants():
    assert "i" in AVAILABILITY_HARD_EXCLUDE_STATUS
    assert "u" in AVAILABILITY_HARD_EXCLUDE_STATUS
    assert "s" in AVAILABILITY_HARD_EXCLUDE_STATUS
    assert "n" in AVAILABILITY_HARD_EXCLUDE_STATUS
    assert 0 in AVAILABILITY_HARD_EXCLUDE_CHANCE
    assert 25 in AVAILABILITY_HARD_EXCLUDE_CHANCE

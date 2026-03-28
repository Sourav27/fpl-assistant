from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path("data")
VAASTAV_DIR = DATA_DIR / "Fantasy-Premier-League"
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")

FPL_API_BASE = "https://fantasy.premierleague.com/api"
FPL_BOOTSTRAP_URL = f"{FPL_API_BASE}/bootstrap-static/"
FPL_PLAYER_URL = f"{FPL_API_BASE}/element-summary"  # /{id}/
FPL_FIXTURES_URL = f"{FPL_API_BASE}/fixtures/"

CURRENT_SEASON = "2025-26"
SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]

# Model config — user promotes a new model by updating this path
ACTIVE_MODEL = MODELS_DIR / "rf_model_gw31.sav"

SQUAD_RULES = {
    "squad_size": 15,
    "xi_size": 11,
    "budget": 1000,  # in 0.1m units
    "max_per_team": 3,
    "positions": {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
    "xi_positions": {"GK": 1, "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)},
}

# Availability filtering constants (hybrid Option C)
AVAILABILITY_HARD_EXCLUDE_STATUS = {"i", "u", "s", "n"}
AVAILABILITY_HARD_EXCLUDE_CHANCE = {0, 25}
AVAILABILITY_SOFT_SCALE = {50: 0.50, 75: 0.75}

# Rate limiting
API_REQUEST_DELAY = 0.5  # seconds between player history fetches
API_RETRY_ATTEMPTS = 3
API_RETRY_BASE_DELAY = 1  # seconds, doubles each attempt

# Bootstrap snapshot staleness threshold
BOOTSTRAP_MAX_AGE_HOURS = 48

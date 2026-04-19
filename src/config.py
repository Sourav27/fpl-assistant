from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path("data")
VAASTAV_DIR = DATA_DIR / "Fantasy-Premier-League"
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")
SNAPSHOTS_DIR = DATA_DIR / "snapshots"


def gw_dir(season: str, gw: int) -> Path:
    return RESULTS_DIR / season / f"gw{gw}"


def snapshot_dir(season: str, gw: int) -> Path:
    return SNAPSHOTS_DIR / season / f"gw{gw}"


# Model registry paths (Track I)
BENCHMARK_PATH      = MODELS_DIR / "benchmark.json"
METRICS_LEDGER_PATH = MODELS_DIR / "metrics_history.jsonl"
CHARTS_DIR          = MODELS_DIR / "charts"

SOURCE_VALIDATION_CSV = RESULTS_DIR / "source_validation.csv"
SIGNAL_ACCURACY_CSV   = RESULTS_DIR / "signal_accuracy.csv"
FPL_API_BASE = "https://fantasy.premierleague.com/api"
FPL_BOOTSTRAP_URL = f"{FPL_API_BASE}/bootstrap-static/"
FPL_PLAYER_URL = f"{FPL_API_BASE}/element-summary"  # /{id}/
FPL_FIXTURES_URL = f"{FPL_API_BASE}/fixtures/"
FPL_ENTRY_URL = f"{FPL_API_BASE}/entry"        # /{id}/ → entry info + bank
FPL_EVENT_URL = f"{FPL_API_BASE}/event"         # /{gw}/live/ → live GW scores
FPL_LEAGUES_CLASSIC_URL = f"{FPL_API_BASE}/leagues-classic"  # /{id}/standings/

USER_CONFIG_DEFAULTS = {
    "horizon_gws": 5,
    "max_hit_points": 8,
    "fdr_sensitivity": 0.15,
}


class UserConfigError(ValueError):
    """Raised when user_config.yaml is missing or invalid."""
    pass


def load_user_config(path: Path | None = None) -> dict:
    """Load and validate user_config.yaml.

    Returns config dict with defaults applied for missing preference keys.
    Raises UserConfigError for missing file or invalid values.
    """
    if path is None:
        path = PROJECT_ROOT / "user_config.yaml"

    if not path.exists():
        example = path.parent / "user_config.example.yaml"
        raise UserConfigError(
            f"user_config.yaml not found at {path}. "
            f"Copy {example} and fill in your entry_id."
        )

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    # Validate required: teams.default.entry_id
    teams = cfg.get("teams", {})
    default_team = teams.get("default", {})
    entry_id = default_team.get("entry_id")
    if entry_id is None:
        raise UserConfigError("user_config.yaml: teams.default.entry_id is required")
    if not isinstance(entry_id, int):
        raise UserConfigError(
            f"user_config.yaml: teams.default.entry_id must be an integer, got {entry_id!r}"
        )

    # Validate alt team if present
    alt_team = teams.get("alt", {})
    if alt_team and "entry_id" in alt_team:
        if not isinstance(alt_team["entry_id"], int):
            raise UserConfigError("user_config.yaml: teams.alt.entry_id must be an integer")

    # Apply defaults for preferences
    prefs = cfg.get("preferences", {})
    for key, default in USER_CONFIG_DEFAULTS.items():
        prefs.setdefault(key, default)
    cfg["preferences"] = prefs

    # Validate horizon_gws
    horizon = prefs["horizon_gws"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not (1 <= horizon <= 5):
        raise UserConfigError(
            f"user_config.yaml: horizon_gws must be an integer 1-5, got {horizon!r}"
        )

    return cfg


CURRENT_SEASON = "2025-26"
SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]

# Model config — user promotes a new model by updating this path
ACTIVE_MODEL = MODELS_DIR / "rf_model_gw31.sav"

# Per-position model paths (Track B). Keys must match ELEMENT_TYPE_MAP values.
ACTIVE_MODELS = {
    "GK":  MODELS_DIR / "rf_gk_gw31.sav",
    "DEF": MODELS_DIR / "rf_def_gw31.sav",
    "MID": MODELS_DIR / "rf_mid_gw31.sav",
    "FWD": MODELS_DIR / "rf_fwd_gw31.sav",
}

def get_active_models() -> dict:
    """Return the active per-position model paths.

    Checks for models/active_models.json at call time (not import time).
    Falls back to the hardcoded ACTIVE_MODELS dict if the manifest is absent
    or malformed. Use this instead of referencing ACTIVE_MODELS directly in
    predict.py and run.py to pick up promotions without restarting.
    """
    import json as _json

    manifest_path = MODELS_DIR / "active_models.json"
    if manifest_path.exists():
        try:
            data = _json.loads(manifest_path.read_text())
            return {pos: MODELS_DIR / info["file"]
                    for pos, info in data.get("models", {}).items()}
        except Exception:
            pass
    return dict(ACTIVE_MODELS)


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

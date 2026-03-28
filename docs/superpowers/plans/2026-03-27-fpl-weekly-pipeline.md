# FPL Weekly Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully automated weekly FPL prediction pipeline that fetches live data from the FPL API, generates predictions with availability filtering, and optimizes team selection — scheduled around the GW32-38 FPL calendar (Apr 10 – May 24, 2026).

**Architecture:** Python-only pipeline (replacing R optimization). Four scheduled phases per gameweek: (1) pre-deadline data fetch + xP capture, (2) post-deadline predictions + optimization, (3) post-GW results collection + live data patching, (4) manual retrain (on-demand). Core modules: `src/pipeline/` for orchestration, `src/data_collection/` (existing, enhanced), `tests/` for TDD.

**Tech Stack:** Python 3.12, pandas, scikit-learn, xgboost, PuLP (LP optimizer), pytest, requests

---

## Design Decisions (from spec)

### Data Combination — Vaastav Base + Live API Patch

| Layer | Source | Coverage | Update Cadence |
|-------|--------|----------|----------------|
| Historical seasons | vaastav `merged_gw.csv` | 2016-17 through 2024-25 | Static (complete) |
| Current season base | vaastav `merged_gw.csv` | 2025-26 GW1-29 | Updated ~3x/season |
| Current season live | FPL API `element-summary/{id}/` | 2025-26 GW30+ | After each GW |

- Deduplication: prefer vaastav over live (richer columns); only use `_live.csv` for GWs not in vaastav
- Live files saved as `data/Fantasy-Premier-League/data/2025-26/gws/gw{N}_live.csv`
- Column categories: Directly mapped (24 cols), Derived post-fetch (6 cols), Unavailable from API (4 cols — fill NaN)
- Bootstrap `elements` list is authoritative roster — exclude departed players, use `ep_this`/`ep_next` fallback for new signings

### Player Availability Filtering — Hybrid (Option C)

Decision table (first match wins):

| # | `status` | `chance_of_playing_next_round` | Action |
|---|----------|-------------------------------|--------|
| 1 | `i`, `u`, `s`, `n` | any | **Hard exclude** |
| 2 | any | `0` or `25` | **Hard exclude** |
| 3 | any | `50` | **Soft scale: xP * 0.50** |
| 4 | `d` | `null` | **Soft scale: xP * 0.50** |
| 5 | any | `75` | **Soft scale: xP * 0.75** |
| 6 | `a` | `100` or `null` | **No adjustment** |
| 7 | `d` | `100` | **No adjustment** |

### Model Retraining — Static with Manual Retrain

- RF model stays frozen during weekly runs; `retrain` CLI phase available on-demand
- RF-only scope (no positional models, no XGBoost for 7-GW timeline)
- New model saved as `rf_model_gw{N}.sav`; user promotes by updating `ACTIVE_MODEL` in config

### API Failure Handling

- Exponential backoff (3 retries, 1s/2s/4s delays) on all API calls
- `post-gw` failure: skip live data collection, log warning, fall back to last dataset
- `pre-deadline` failure: skip xP capture, use model's predicted xP, skip availability filtering
- Cached bootstrap snapshots in `results/snapshots/` — use if < 48 hours old

### Rate Limiting (post-gw player fetches)

- Fetch only players present in current bootstrap (~500-600 active)
- 0.5s sleep between requests (~5 minutes total)
- Log progress every 50 players

---

## FPL Calendar — Remaining Gameweeks

| GW | Deadline (UTC) | Notes |
|----|----------------|-------|
| 32 | Apr 10, 2026 17:30 | **First target** — 2 weeks from now |
| 33 | Apr 18, 2026 10:00 | |
| 34 | Apr 24, 2026 17:30 | |
| 35 | May 2, 2026 12:30 | Weekly cadence starts |
| 36 | May 9, 2026 12:30 | |
| 37 | May 17, 2026 12:30 | |
| 38 | May 24, 2026 13:30 | Season finale |

---

## File Structure

### New files to create

```
fpl-assistant/
├── pyproject.toml                          # Project config, pytest settings
├── src/
│   ├── __init__.py
│   ├── config.py                           # Paths, seasons, API URLs, constants, ACTIVE_MODEL
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── fetch.py                        # FPL API data fetching (bootstrap, players, fixtures, bulk history)
│   │   ├── prepare.py                      # Build merged dataset from vaastav + live data with dedup
│   │   ├── features.py                     # Vectorized feature engineering (replaces NB02 iterrows)
│   │   ├── predict.py                      # Load models, generate predictions for next GW
│   │   ├── availability.py                 # Player availability filtering (hybrid hard/soft)
│   │   ├── optimize.py                     # PuLP-based team selection (replaces R scripts)
│   │   └── run.py                          # CLI entry point: orchestrates phases
│   └── data_collection/                    # Existing — enhanced, not replaced
│       └── (existing files stay)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                         # Shared fixtures: sample DataFrames, mock API responses
│   ├── test_config.py
│   ├── test_fetch.py
│   ├── test_prepare.py
│   ├── test_features.py
│   ├── test_predict.py
│   ├── test_availability.py
│   ├── test_optimize.py
│   ├── test_run.py
│   └── test_integration.py
└── scripts/
    └── weekly_run.sh                       # Cron-friendly wrapper script
```

### Existing files to modify

```
src/data_collection/gameweek.py             # Fix deprecated datetime.utcnow()
src/data_collection/getters.py              # Add resume capability, timeout handling
requirements.txt                            # Add pytest, PuLP
```

---

## Task 1: Project Setup and Test Infrastructure

**Files:**
- Create: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `tests/test_config.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add test dependencies to requirements.txt**

```
# Testing
pytest>=7.0.0
pytest-mock>=3.10.0

# Optimization (Python replacement for R lpSolve)
PuLP>=2.7.0
```

Append these lines to the end of `requirements.txt`.

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "fpl-assistant"
version = "0.1.0"
requires-python = ">=3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-v --tb=short"
```

- [ ] **Step 3: Write failing test for config module**

```python
# tests/test_config.py
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
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 5: Create src/__init__.py and src/config.py**

```python
# src/__init__.py
```

```python
# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VAASTAV_DIR = DATA_DIR / "Fantasy-Premier-League"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

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
ACTIVE_MODEL = MODELS_DIR / "rf_model.sav"

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
```

- [ ] **Step 6: Create tests/__init__.py and tests/conftest.py**

```python
# tests/__init__.py
```

```python
# tests/conftest.py
import pytest
import pandas as pd
import json

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
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: All 6 tests PASS

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml requirements.txt src/__init__.py src/config.py tests/
git commit -m "feat: add project config, test infrastructure, and shared fixtures"
```

---

## Task 2: FPL API Data Fetcher (with Live Data Collection)

**Files:**
- Create: `src/pipeline/__init__.py`
- Create: `src/pipeline/fetch.py`
- Create: `tests/test_fetch.py`

- [ ] **Step 1: Write failing tests for fetch module**

```python
# tests/test_fetch.py
import json
import pytest
import requests
from unittest.mock import patch, MagicMock, call
from src.pipeline.fetch import (
    fetch_bootstrap,
    fetch_player_history,
    fetch_fixtures,
    get_current_gw,
    get_next_deadline,
    extract_xp_snapshot,
    normalize_player_gw_to_vaastav,
    fetch_live_gw_data,
)


class TestFetchBootstrap:
    def test_returns_dict_with_expected_keys(self, sample_bootstrap_json):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_bootstrap_json

        with patch("src.pipeline.fetch.requests.get", return_value=mock_resp):
            data = fetch_bootstrap()

        assert "elements" in data
        assert "events" in data
        assert "teams" in data

    def test_raises_on_http_error_after_retries(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.raise_for_status.side_effect = requests.RequestException("503 Server Error")

        with patch("src.pipeline.fetch.requests.get", return_value=mock_resp), \
             patch("src.pipeline.fetch.time.sleep"):  # skip retry delays
            with pytest.raises(requests.RequestException, match="503"):
                fetch_bootstrap()


class TestGetCurrentGW:
    def test_identifies_current_gw(self, sample_bootstrap_json):
        result = get_current_gw(sample_bootstrap_json)
        assert result == 30

    def test_returns_none_when_no_current(self):
        data = {"events": [{"id": 1, "is_current": False, "is_next": False}]}
        result = get_current_gw(data)
        assert result is None


class TestGetNextDeadline:
    def test_returns_next_deadline(self, sample_bootstrap_json):
        gw, deadline = get_next_deadline(sample_bootstrap_json)
        assert gw == 31
        assert "2026-03-20" in deadline


class TestExtractXpSnapshot:
    def test_returns_dict_of_id_to_xp(self, sample_bootstrap_json):
        result = extract_xp_snapshot(sample_bootstrap_json)
        assert result[1] == 4.2
        assert result[3] == 6.8

    def test_handles_none_ep_this(self):
        data = {"elements": [{"id": 1, "ep_this": None}]}
        result = extract_xp_snapshot(data)
        assert result[1] == 0.0


class TestFetchPlayerHistory:
    def test_returns_history_list(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "history": [{"round": 1, "total_points": 5}],
            "history_past": [],
        }

        with patch("src.pipeline.fetch.requests.get", return_value=mock_resp):
            data = fetch_player_history(1)
        assert len(data["history"]) == 1


class TestFetchFixtures:
    def test_returns_list(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"id": 1, "event": 30, "team_h": 1, "team_a": 10,
             "team_h_difficulty": 3, "team_a_difficulty": 4}
        ]

        with patch("src.pipeline.fetch.requests.get", return_value=mock_resp):
            data = fetch_fixtures()
        assert len(data) == 1
        assert data[0]["team_h_difficulty"] == 3


class TestNormalizePlayerGwToVaastav:
    def test_maps_api_fields_to_vaastav_schema(self, sample_bootstrap_json, sample_player_history_json):
        gw_row = sample_player_history_json["history"][0]
        bootstrap = sample_bootstrap_json

        result = normalize_player_gw_to_vaastav(gw_row, bootstrap)

        assert result["name"] == "Saka"
        assert result["position"] == "MID"
        assert result["team"] == "Arsenal"
        assert result["total_points"] == 8
        assert result["GW"] == 1
        assert result["season"] == "2025-26"
        assert result["element"] == 3

    def test_fills_unavailable_columns_with_nan(self, sample_bootstrap_json, sample_player_history_json):
        import math
        gw_row = sample_player_history_json["history"][0]
        result = normalize_player_gw_to_vaastav(gw_row, sample_bootstrap_json)

        # These columns are unavailable from API — must exist and be NaN
        assert "clearances_blocks_interceptions" in result
        assert math.isnan(result["clearances_blocks_interceptions"])
        assert "defensive_contribution" in result
        assert math.isnan(result["defensive_contribution"])


class TestFetchLiveGwData:
    def test_returns_dataframe_for_one_gw(self, sample_bootstrap_json, sample_player_history_json):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_player_history_json

        # Only fetch for element IDs that exist in bootstrap
        with patch("src.pipeline.fetch.requests.get", return_value=mock_resp):
            with patch("src.pipeline.fetch.time.sleep"):  # skip rate limit delay
                result = fetch_live_gw_data(
                    target_gw=1,
                    bootstrap_data=sample_bootstrap_json,
                    player_ids=[3],  # just Saka
                )

        assert len(result) == 1
        assert result.iloc[0]["name"] == "Saka"
        assert result.iloc[0]["GW"] == 1

    def test_skips_players_without_gw_data(self, sample_bootstrap_json):
        """Player has no history for the target GW."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"history": [], "history_past": [], "fixtures": []}

        with patch("src.pipeline.fetch.requests.get", return_value=mock_resp):
            with patch("src.pipeline.fetch.time.sleep"):
                result = fetch_live_gw_data(
                    target_gw=30,
                    bootstrap_data=sample_bootstrap_json,
                    player_ids=[3],
                )

        assert len(result) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/pipeline/__init__.py
```

```python
# src/pipeline/fetch.py
"""FPL API data fetching — thin wrappers with retry logic and live data collection."""
import time
import logging
import pandas as pd
import requests
from src.config import (
    FPL_BOOTSTRAP_URL, FPL_PLAYER_URL, FPL_FIXTURES_URL,
    CURRENT_SEASON, API_REQUEST_DELAY, API_RETRY_ATTEMPTS, API_RETRY_BASE_DELAY,
)

logger = logging.getLogger(__name__)

# element_type ID → position string (API uses "GKP" but we normalize to "GK" for vaastav compat)
ELEMENT_TYPE_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Columns that exist in vaastav but cannot be derived from API
UNAVAILABLE_FROM_API = [
    "clearances_blocks_interceptions", "defensive_contribution",
    "recoveries", "tackles",
]


def _api_get_with_retry(url: str, timeout: int = 30) -> requests.Response:
    """GET with exponential backoff retry."""
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == API_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(API_RETRY_BASE_DELAY * (2 ** attempt))


def fetch_bootstrap() -> dict:
    """Fetch the main FPL bootstrap-static endpoint."""
    return _api_get_with_retry(FPL_BOOTSTRAP_URL).json()


def fetch_player_history(player_id: int) -> dict:
    """Fetch individual player GW history + past seasons."""
    url = f"{FPL_PLAYER_URL}/{player_id}/"
    return _api_get_with_retry(url, timeout=15).json()


def fetch_fixtures() -> list[dict]:
    """Fetch all fixtures for the season."""
    return _api_get_with_retry(FPL_FIXTURES_URL).json()


def get_current_gw(bootstrap_data: dict) -> int | None:
    """Return the current gameweek number from bootstrap data."""
    for event in bootstrap_data["events"]:
        if event["is_current"]:
            return event["id"]
    return None


def get_next_deadline(bootstrap_data: dict) -> tuple[int, str]:
    """Return (gw_number, deadline_time) for the next upcoming GW."""
    for event in bootstrap_data["events"]:
        if event["is_next"]:
            return event["id"], event["deadline_time"]
    raise ValueError("No upcoming gameweek found")


def extract_xp_snapshot(bootstrap_data: dict) -> dict[int, float]:
    """Extract {player_id: ep_this} from bootstrap data.

    Must be called BEFORE the GW deadline — ep_this is forward-looking.
    """
    result = {}
    for el in bootstrap_data["elements"]:
        ep = el.get("ep_this")
        result[el["id"]] = float(ep) if ep is not None else 0.0
    return result


def _build_bootstrap_lookups(bootstrap_data: dict) -> tuple[dict, dict, dict]:
    """Build lookup dicts from bootstrap data for normalization."""
    # team_id → team_name
    team_map = {t["id"]: t["name"] for t in bootstrap_data["teams"]}
    # element_id → element info
    element_map = {e["id"]: e for e in bootstrap_data["elements"]}
    # element_type_id → position string
    pos_map = ELEMENT_TYPE_MAP.copy()
    return team_map, element_map, pos_map


def normalize_player_gw_to_vaastav(
    gw_row: dict,
    bootstrap_data: dict,
    _lookups: tuple | None = None,
) -> dict:
    """Normalize a single API player-GW history row to vaastav schema."""
    team_map, element_map, pos_map = _lookups or _build_bootstrap_lookups(bootstrap_data)
    element_id = gw_row["element"]
    element_info = element_map.get(element_id, {})

    row = {
        # Derived fields
        "name": element_info.get("web_name", "Unknown"),
        "position": pos_map.get(element_info.get("element_type"), "UNK"),
        "team": team_map.get(element_info.get("team"), "Unknown"),
        "element": element_id,
        "GW": gw_row["round"],
        "season": CURRENT_SEASON,
        "xP": 0.0,  # filled later from xP snapshot if available
        # Directly mapped fields
        "total_points": gw_row.get("total_points", 0),
        "minutes": gw_row.get("minutes", 0),
        "goals_scored": gw_row.get("goals_scored", 0),
        "assists": gw_row.get("assists", 0),
        "clean_sheets": gw_row.get("clean_sheets", 0),
        "goals_conceded": gw_row.get("goals_conceded", 0),
        "bonus": gw_row.get("bonus", 0),
        "bps": gw_row.get("bps", 0),
        "influence": float(gw_row.get("influence", 0)),
        "creativity": float(gw_row.get("creativity", 0)),
        "threat": float(gw_row.get("threat", 0)),
        "ict_index": float(gw_row.get("ict_index", 0)),
        "value": gw_row.get("value", 0),
        "transfers_in": gw_row.get("transfers_in", 0),
        "transfers_out": gw_row.get("transfers_out", 0),
        "selected": gw_row.get("selected", 0),
        "was_home": gw_row.get("was_home", False),
        "opponent_team": gw_row.get("opponent_team", 0),
        "fixture": gw_row.get("fixture", 0),
        "round": gw_row.get("round", 0),
        "kickoff_time": gw_row.get("kickoff_time", ""),
        "starts": gw_row.get("starts", 0),
        "expected_goals": float(gw_row.get("expected_goals", 0)),
        "expected_assists": float(gw_row.get("expected_assists", 0)),
        "expected_goal_involvements": float(gw_row.get("expected_goal_involvements", 0)),
        "expected_goals_conceded": float(gw_row.get("expected_goals_conceded", 0)),
    }

    # Unavailable from API — fill with NaN
    for col in UNAVAILABLE_FROM_API:
        row[col] = float("nan")

    return row


def fetch_live_gw_data(
    target_gw: int,
    bootstrap_data: dict,
    player_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Fetch all player data for a specific GW and normalize to vaastav schema.

    Args:
        target_gw: The GW number to collect data for.
        bootstrap_data: Bootstrap-static response (for lookups).
        player_ids: Optional subset of player IDs. Defaults to all active players.
    """
    if player_ids is None:
        player_ids = [e["id"] for e in bootstrap_data["elements"]]

    lookups = _build_bootstrap_lookups(bootstrap_data)

    rows = []
    for i, pid in enumerate(player_ids):
        if i > 0:
            time.sleep(API_REQUEST_DELAY)
        if (i + 1) % 50 == 0:
            logger.info(f"Fetching player {i + 1}/{len(player_ids)}")

        try:
            data = fetch_player_history(pid)
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch player {pid}: {e}")
            continue

        for gw_row in data.get("history", []):
            if gw_row["round"] == target_gw:
                normalized = normalize_player_gw_to_vaastav(gw_row, bootstrap_data, _lookups=lookups)
                rows.append(normalized)
                break

    return pd.DataFrame(rows) if rows else pd.DataFrame()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fetch.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/ tests/test_fetch.py
git commit -m "feat: add FPL API fetch module with live data collection and schema normalization"
```

---

## Task 3: Data Preparation Module (with Live Data Merge)

**Files:**
- Create: `src/pipeline/prepare.py`
- Create: `tests/test_prepare.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_prepare.py
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch
from src.pipeline.prepare import (
    load_season_gw_data,
    load_live_gw_files,
    merge_seasons,
    add_fixture_difficulty,
    build_merged_dataset,
)


class TestLoadSeasonGwData:
    def test_loads_csv_with_season_column(self, tmp_path):
        gw_dir = tmp_path / "data" / "Fantasy-Premier-League" / "data" / "2024-25" / "gws"
        gw_dir.mkdir(parents=True)
        df = pd.DataFrame({"name": ["Saka"], "total_points": [8], "GW": [1]})
        df.to_csv(gw_dir / "merged_gw.csv", index=False)

        result = load_season_gw_data("2024-25", vaastav_dir=tmp_path / "data" / "Fantasy-Premier-League")
        assert "season" in result.columns
        assert result["season"].iloc[0] == "2024-25"

    def test_handles_latin1_encoding(self, tmp_path):
        gw_dir = tmp_path / "data" / "Fantasy-Premier-League" / "data" / "2017-18" / "gws"
        gw_dir.mkdir(parents=True)
        df = pd.DataFrame({"name": ["Agüero"], "total_points": [10], "GW": [1]})
        df.to_csv(gw_dir / "merged_gw.csv", index=False, encoding="latin-1")

        result = load_season_gw_data("2017-18", vaastav_dir=tmp_path / "data" / "Fantasy-Premier-League")
        assert result["name"].iloc[0] == "Agüero"


class TestLoadLiveGwFiles:
    def test_discovers_live_csv_files(self, tmp_path):
        gw_dir = tmp_path / "gws"
        gw_dir.mkdir()
        df = pd.DataFrame({"name": ["Saka"], "total_points": [8], "GW": [30], "element": [3]})
        df.to_csv(gw_dir / "gw30_live.csv", index=False)
        df2 = pd.DataFrame({"name": ["Saka"], "total_points": [6], "GW": [31], "element": [3]})
        df2.to_csv(gw_dir / "gw31_live.csv", index=False)

        result = load_live_gw_files(gw_dir)
        assert len(result) == 2

    def test_returns_empty_when_no_live_files(self, tmp_path):
        gw_dir = tmp_path / "gws"
        gw_dir.mkdir()
        result = load_live_gw_files(gw_dir)
        assert len(result) == 0

    def test_dedup_prefers_vaastav_over_live(self, tmp_path):
        """When vaastav merged_gw.csv covers a GW, live data is dropped."""
        gw_dir = tmp_path / "data" / "FPL" / "data" / "2025-26" / "gws"
        gw_dir.mkdir(parents=True)

        # vaastav data covers GW30
        vaastav_df = pd.DataFrame({
            "name": ["Saka"], "total_points": [8], "GW": [30],
            "element": [3], "tackles": [2],  # richer columns
        })
        vaastav_df.to_csv(gw_dir / "merged_gw.csv", index=False)

        # live data also has GW30
        live_df = pd.DataFrame({
            "name": ["Saka"], "total_points": [8], "GW": [30], "element": [3],
        })
        live_df.to_csv(gw_dir / "gw30_live.csv", index=False)

        # live GW31 not in vaastav
        live_df2 = pd.DataFrame({
            "name": ["Saka"], "total_points": [6], "GW": [31], "element": [3],
        })
        live_df2.to_csv(gw_dir / "gw31_live.csv", index=False)

        result = build_merged_dataset(
            seasons=["2025-26"],
            vaastav_dir=tmp_path / "data" / "FPL",
        )
        # Should have GW30 from vaastav (with tackles) and GW31 from live
        assert len(result) == 2
        gw30 = result[result["GW"] == 30]
        assert "tackles" in result.columns
        assert gw30.iloc[0]["tackles"] == 2  # vaastav row, not live


class TestMergeSeasons:
    def test_concatenates_with_common_columns(self):
        df1 = pd.DataFrame({"name": ["A"], "total_points": [5], "season": ["2023-24"]})
        df2 = pd.DataFrame({"name": ["B"], "total_points": [3], "season": ["2024-25"], "tackles": [2]})
        result = merge_seasons([df1, df2])
        assert len(result) == 2
        assert "tackles" in result.columns  # schema union, not intersection


class TestAddFixtureDifficulty:
    def test_adds_fdr_columns(self, tmp_path):
        fixtures_path = tmp_path / "fixtures.csv"
        fixtures = pd.DataFrame({
            "id": [1], "event": [1], "team_h": [1], "team_a": [10],
            "team_h_difficulty": [3], "team_a_difficulty": [4],
        })
        fixtures.to_csv(fixtures_path, index=False)

        gw_df = pd.DataFrame({
            "fixture": [1], "was_home": [True], "season": ["2025-26"],
        })
        result = add_fixture_difficulty(gw_df, fixtures_path)
        assert "fdr_team" in result.columns
        assert result["fdr_team"].iloc[0] == 3


class TestBuildMergedDataset:
    def test_end_to_end_produces_expected_columns(self, tmp_path):
        season_dir = tmp_path / "FPL" / "data" / "2025-26"
        gw_dir = season_dir / "gws"
        gw_dir.mkdir(parents=True)

        gw_data = pd.DataFrame({
            "name": ["Saka", "Saka"], "position": ["MID", "MID"],
            "team": ["Arsenal", "Arsenal"], "element": [3, 3],
            "total_points": [8, 6], "minutes": [90, 90],
            "fixture": [1, 2], "was_home": [True, False],
            "GW": [1, 2], "xP": [6.5, 5.0],
            "goals_scored": [1, 0], "assists": [1, 0],
            "clean_sheets": [0, 0], "ict_index": [12.0, 5.0],
            "influence": [40.0, 15.0], "creativity": [35.0, 10.0],
            "threat": [50.0, 20.0], "bps": [35, 12], "bonus": [3, 0],
            "value": [105, 105], "transfers_in": [5000, 3000],
            "transfers_out": [1000, 2000], "selected": [3000000, 3100000],
            "opponent_team": [10, 15], "round": [1, 2],
        })
        gw_data.to_csv(gw_dir / "merged_gw.csv", index=False)

        fixtures = pd.DataFrame({
            "id": [1, 2], "event": [1, 2],
            "team_h": [1, 10], "team_a": [10, 1],
            "team_h_difficulty": [3, 4], "team_a_difficulty": [4, 3],
        })
        fixtures.to_csv(season_dir / "fixtures.csv", index=False)

        result = build_merged_dataset(
            seasons=["2025-26"],
            vaastav_dir=tmp_path / "FPL",
        )
        assert len(result) == 2
        assert "season" in result.columns
        assert "fdr_team" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prepare.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/pipeline/prepare.py
"""Build the merged multi-season dataset from vaastav data + live API patches."""
import pandas as pd
from pathlib import Path
from src.config import VAASTAV_DIR, SEASONS, CURRENT_SEASON


LATIN1_SEASONS = {"2016-17", "2017-18", "2018-19"}


def load_season_gw_data(season: str, vaastav_dir: Path = VAASTAV_DIR) -> pd.DataFrame:
    """Load merged_gw.csv for a single season."""
    path = vaastav_dir / "data" / season / "gws" / "merged_gw.csv"
    encoding = "latin-1" if season in LATIN1_SEASONS else "utf-8"
    df = pd.read_csv(path, encoding=encoding, low_memory=False)
    df["season"] = season
    return df


def load_live_gw_files(gw_dir: Path) -> pd.DataFrame:
    """Load all gw{N}_live.csv files from a directory."""
    live_files = sorted(gw_dir.glob("gw*_live.csv"))
    if not live_files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in live_files]
    return pd.concat(dfs, ignore_index=True, sort=False)


def merge_seasons(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate season DataFrames, taking the union of all columns."""
    return pd.concat(dfs, ignore_index=True, sort=False)


def add_fixture_difficulty(gw_df: pd.DataFrame, fixtures_path: Path) -> pd.DataFrame:
    """Join FDR ratings from fixtures.csv onto gameweek data."""
    fixtures = pd.read_csv(fixtures_path)
    fixtures = fixtures[["id", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]]
    fixtures = fixtures.rename(columns={"id": "fixture"})

    df = gw_df.merge(fixtures, on="fixture", how="left")

    import numpy as np
    home = df["was_home"].astype(bool)
    df["fdr_team"] = np.where(home, df["team_h_difficulty"], df["team_a_difficulty"])
    df["fdr_opp"] = np.where(home, df["team_a_difficulty"], df["team_h_difficulty"])
    df = df.drop(columns=["team_h", "team_a", "team_h_difficulty", "team_a_difficulty"])
    return df


def build_merged_dataset(
    seasons: list[str] | None = None,
    vaastav_dir: Path = VAASTAV_DIR,
) -> pd.DataFrame:
    """Build the full merged dataset: vaastav base + live API patches.

    Deduplication: prefer vaastav over live (richer columns).
    Live data only used for GWs not covered in vaastav's merged_gw.csv.
    """
    seasons = seasons or SEASONS
    dfs = []
    for season in seasons:
        path = vaastav_dir / "data" / season / "gws" / "merged_gw.csv"
        if not path.exists():
            continue
        df = load_season_gw_data(season, vaastav_dir)

        # Load live patches for current season only
        gw_dir = vaastav_dir / "data" / season / "gws"
        live_df = load_live_gw_files(gw_dir) if season == CURRENT_SEASON else pd.DataFrame()
        if not live_df.empty:
            # Only keep live rows for GWs not already in vaastav
            vaastav_gws = set(df["GW"].unique()) if "GW" in df.columns else set()
            live_df = live_df[~live_df["GW"].isin(vaastav_gws)]
            if not live_df.empty:
                live_df["season"] = season
                df = pd.concat([df, live_df], ignore_index=True, sort=False)

        fixtures_path = vaastav_dir / "data" / season / "fixtures.csv"
        if fixtures_path.exists():
            df = add_fixture_difficulty(df, fixtures_path)
        dfs.append(df)

    return merge_seasons(dfs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prepare.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/prepare.py tests/test_prepare.py
git commit -m "feat: add data preparation module with vaastav+live merge and dedup"
```

---

## Task 4: Vectorized Feature Engineering

**Files:**
- Create: `src/pipeline/features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# src/pipeline/features.py
"""Vectorized feature engineering — replaces NB02 iterrows approach."""
import pandas as pd
import numpy as np

ROLLING_COLS = [
    "total_points", "minutes", "ict_index", "bps",
    "goals_scored", "assists", "clean_sheets",
    "influence", "creativity", "threat",
]

DEFAULT_WINDOWS = [4, 8]


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add lagged rolling averages per player.

    Uses shift(1) so GW N's features are computed from GW 1..(N-1).
    """
    windows = windows or DEFAULT_WINDOWS
    cols = cols or [c for c in ROLLING_COLS if c in df.columns]
    df = df.sort_values(["element", "season", "GW"]).copy()

    for col in cols:
        for w in windows:
            df[f"{col}_roll_{w}"] = (
                df.groupby("element")[col]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=w).mean())
            )
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum = short-term rolling avg - long-term rolling avg."""
    for col in ROLLING_COLS:
        short = f"{col}_roll_4"
        long = f"{col}_roll_8"
        if short in df.columns and long in df.columns:
            df[f"{col}_momentum"] = df[short] - df[long]
    return df


def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived form features."""
    if "transfers_in" in df.columns and "transfers_out" in df.columns:
        df["transfers_net"] = df["transfers_in"] - df["transfers_out"]
    return df


def engineer_features(
    df: pd.DataFrame,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_rolling_features(df)
    df = add_momentum_features(df)
    df = add_form_features(df)
    if drop_na:
        longest_window = max(DEFAULT_WINDOWS)
        df = df.dropna(subset=[f"total_points_roll_{longest_window}"])
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/features.py tests/test_features.py
git commit -m "feat: add vectorized feature engineering with rolling averages and momentum"
```

---

## Task 5: Prediction Module

**Files:**
- Create: `src/pipeline/predict.py`
- Create: `tests/test_predict.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_predict.py
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.pipeline.predict import (
    load_model,
    get_feature_columns,
    predict_next_gw,
)


@pytest.fixture
def trained_model(tmp_path):
    """Create a mock model .sav file."""
    from sklearn.ensemble import RandomForestRegressor
    import joblib

    model = RandomForestRegressor(n_estimators=5, random_state=42)
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    model.fit(X, y)
    model_path = tmp_path / "rf_model.sav"
    joblib.dump(model, model_path)
    return model_path, model.feature_names_in_ if hasattr(model, "feature_names_in_") else None


class TestLoadModel:
    def test_loads_saved_model(self, trained_model):
        model_path, _ = trained_model
        model = load_model(model_path)
        assert hasattr(model, "predict")

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.sav")


class TestGetFeatureColumns:
    def test_returns_expected_feature_set(self):
        cols = get_feature_columns()
        assert "total_points_roll_4" in cols
        assert "minutes_roll_4" in cols
        assert "transfers_net" in cols
        # Should NOT include target or identifiers
        assert "total_points" not in cols
        assert "name" not in cols
        assert "element" not in cols


class TestPredictNextGW:
    def test_returns_dataframe_with_xp_column(self, trained_model):
        model_path, _ = trained_model
        from sklearn.ensemble import RandomForestRegressor
        import joblib

        # Create model that expects named features
        feature_cols = get_feature_columns()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        X = pd.DataFrame(np.random.rand(50, len(feature_cols)), columns=feature_cols)
        y = np.random.rand(50)
        model.fit(X, y)
        named_model_path = model_path.parent / "rf_named.sav"
        joblib.dump(model, named_model_path)

        player_features = X.copy()
        player_features["element"] = range(50)
        player_features["name"] = [f"Player_{i}" for i in range(50)]
        player_features["position"] = ["MID"] * 50
        player_features["team"] = ["Arsenal"] * 50
        player_features["now_cost"] = [100] * 50

        result = predict_next_gw(player_features, named_model_path)
        assert "xP" in result.columns
        assert "element" in result.columns
        assert len(result) == 50
        assert (result["xP"] >= 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_predict.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# src/pipeline/predict.py
"""Load trained models and generate predictions."""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import ACTIVE_MODEL


FEATURE_COLUMNS = [
    "total_points_roll_4", "total_points_roll_8",
    "minutes_roll_4", "minutes_roll_8",
    "ict_index_roll_4", "ict_index_roll_8",
    "bps_roll_4", "bps_roll_8",
    "goals_scored_roll_4", "assists_roll_4",
    "clean_sheets_roll_4",
    "influence_roll_4", "creativity_roll_4", "threat_roll_4",
    "total_points_momentum", "minutes_momentum",
    "ict_index_momentum",
    "transfers_net",
]

ID_COLUMNS = ["element", "name", "position", "team", "now_cost"]


def load_model(path: Path):
    """Load a joblib-serialized model."""
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def get_feature_columns() -> list[str]:
    """Return the list of feature columns expected by models."""
    return FEATURE_COLUMNS.copy()


def predict_next_gw(
    player_features: pd.DataFrame,
    model_path: Path = ACTIVE_MODEL,
) -> pd.DataFrame:
    """Generate xP predictions for the next gameweek."""
    model = load_model(model_path)
    feature_cols = get_feature_columns()

    # Normalize cost column: vaastav uses 'value', API uses 'now_cost'
    df = player_features.copy()
    if "now_cost" not in df.columns and "value" in df.columns:
        df["now_cost"] = df["value"]

    X = df[feature_cols].copy()
    X = X.fillna(0)

    predictions = model.predict(X)
    predictions = np.clip(predictions, 0, None)  # xP can't be negative

    result = df[ID_COLUMNS].copy()
    result["xP"] = predictions
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_predict.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/predict.py tests/test_predict.py
git commit -m "feat: add prediction module with model loading and xP generation"
```

---

## Task 6: Player Availability Filtering

**Files:**
- Create: `src/pipeline/availability.py`
- Create: `tests/test_availability.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_availability.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/pipeline/availability.py
"""Player availability filtering — hybrid hard-exclude + soft-scale approach."""
import logging
import pandas as pd
from src.config import (
    AVAILABILITY_HARD_EXCLUDE_STATUS,
    AVAILABILITY_HARD_EXCLUDE_CHANCE,
    AVAILABILITY_SOFT_SCALE,
)

logger = logging.getLogger(__name__)


def filter_availability(
    predictions: pd.DataFrame,
    bootstrap_data: dict,
) -> pd.DataFrame:
    """Filter and adjust predictions based on player availability.

    Decision table (first match wins):
      1. status in {i, u, s, n}           → hard exclude
      2. chance in {0, 25}                 → hard exclude
      3. chance == 50                      → xP * 0.50
      4. status == 'd' and chance is None  → xP * 0.50
      5. chance == 75                      → xP * 0.75
      6. status == 'a', chance 100/None    → no adjustment
      7. status == 'd', chance == 100      → no adjustment
    """
    # Build lookup: element_id → availability info
    avail_map = {}
    for el in bootstrap_data.get("elements", []):
        avail_map[el["id"]] = {
            "status": el.get("status", "a"),
            "chance": el.get("chance_of_playing_next_round"),
            "news": el.get("news", ""),
        }

    result = predictions.copy()
    exclude_mask = pd.Series(False, index=result.index)
    scale_factors = pd.Series(1.0, index=result.index)

    for idx, row in result.iterrows():
        info = avail_map.get(row["element"])
        if info is None:
            continue

        status = info["status"]
        chance = info["chance"]

        # Rule 1: Hard exclude by status
        if status in AVAILABILITY_HARD_EXCLUDE_STATUS:
            exclude_mask[idx] = True
            logger.info(f"Excluded {row['name']} (status={status}, news={info['news']})")
            continue

        # Rule 2: Hard exclude by chance
        if chance is not None and chance in AVAILABILITY_HARD_EXCLUDE_CHANCE:
            exclude_mask[idx] = True
            logger.info(f"Excluded {row['name']} (chance={chance}%, news={info['news']})")
            continue

        # Rules 3 & 5: Soft scale by chance (50 → 0.50, 75 → 0.75)
        if chance is not None and chance in AVAILABILITY_SOFT_SCALE:
            scale_factors[idx] = AVAILABILITY_SOFT_SCALE[chance]
            logger.info(f"Scaled {row['name']} xP by {AVAILABILITY_SOFT_SCALE[chance]} (chance={chance}%)")
            continue

        # Rule 4: Doubtful with null chance → treat as 50/50
        if status == "d" and chance is None:
            scale_factors[idx] = 0.50
            logger.info(f"Scaled {row['name']} xP by 0.50 (doubtful, chance=null)")
            continue

        # Rules 5-7: No adjustment needed (chance=100/None with status=a/d)

    # Apply exclusions and scaling
    result = result[~exclude_mask].copy()
    scale_factors = scale_factors[~exclude_mask]
    result["xP"] = result["xP"] * scale_factors.values

    return result.reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_availability.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/availability.py tests/test_availability.py
git commit -m "feat: add hybrid player availability filtering with decision table"
```

---

## Task 7: Python Team Optimizer (Replacing R)

**Files:**
- Create: `src/pipeline/optimize.py`
- Create: `tests/test_optimize.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_optimize.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# src/pipeline/optimize.py
"""PuLP-based FPL team optimizer — replaces R lpSolve scripts."""
import pandas as pd
import pulp
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value
from src.config import SQUAD_RULES


def select_squad(players: pd.DataFrame) -> pd.DataFrame:
    """Select optimal 15-player squad using linear programming."""
    prob = LpProblem("FPL_Squad", LpMaximize)
    n = len(players)
    x = [LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Objective: maximize total xP
    prob += lpSum(x[i] * players.iloc[i]["xP"] for i in range(n))

    # Squad size = 15
    prob += lpSum(x) == SQUAD_RULES["squad_size"]

    # Budget constraint
    prob += lpSum(x[i] * players.iloc[i]["now_cost"] for i in range(n)) <= SQUAD_RULES["budget"]

    # Position constraints
    for pos, count in SQUAD_RULES["positions"].items():
        mask = (players["position"] == pos).values
        prob += lpSum(x[i] for i in range(n) if mask[i]) == count

    # Max 3 per team
    for team in players["team"].unique():
        mask = (players["team"] == team).values
        prob += lpSum(x[i] for i in range(n) if mask[i]) <= SQUAD_RULES["max_per_team"]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    selected = [i for i in range(n) if value(x[i]) is not None and value(x[i]) > 0.5]
    return players.iloc[selected].reset_index(drop=True)


def select_xi(squad: pd.DataFrame) -> pd.DataFrame:
    """Select best 11 from a 15-player squad."""
    prob = LpProblem("FPL_XI", LpMaximize)
    n = len(squad)
    x = [LpVariable(f"xi_{i}", cat="Binary") for i in range(n)]

    prob += lpSum(x[i] * squad.iloc[i]["xP"] for i in range(n))

    # Exactly 11
    prob += lpSum(x) == SQUAD_RULES["xi_size"]

    # Exactly 1 GK
    gk_mask = (squad["position"] == "GK").values
    prob += lpSum(x[i] for i in range(n) if gk_mask[i]) == 1

    # DEF: 3-5
    def_mask = (squad["position"] == "DEF").values
    prob += lpSum(x[i] for i in range(n) if def_mask[i]) >= 3
    prob += lpSum(x[i] for i in range(n) if def_mask[i]) <= 5

    # MID: 2-5
    mid_mask = (squad["position"] == "MID").values
    prob += lpSum(x[i] for i in range(n) if mid_mask[i]) >= 2
    prob += lpSum(x[i] for i in range(n) if mid_mask[i]) <= 5

    # FWD: 1-3
    fwd_mask = (squad["position"] == "FWD").values
    prob += lpSum(x[i] for i in range(n) if fwd_mask[i]) >= 1
    prob += lpSum(x[i] for i in range(n) if fwd_mask[i]) <= 3

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    selected = [i for i in range(n) if value(x[i]) is not None and value(x[i]) > 0.5]
    return squad.iloc[selected].reset_index(drop=True)


def select_captain(xi: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Pick captain (highest xP) and vice-captain (second highest)."""
    sorted_xi = xi.sort_values("xP", ascending=False)
    return sorted_xi.iloc[0], sorted_xi.iloc[1]


def optimize_team(players: pd.DataFrame) -> dict:
    """Full optimization pipeline: squad -> XI -> captain."""
    squad = select_squad(players)
    xi = select_xi(squad)
    captain, vice = select_captain(xi)

    return {
        "squad": squad,
        "xi": xi,
        "captain": captain,
        "vice_captain": vice,
        "total_xp": xi["xP"].sum() + captain["xP"],  # captain points doubled
        "bench": squad[~squad["element"].isin(xi["element"])],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_optimize.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/optimize.py tests/test_optimize.py
git commit -m "feat: add PuLP-based team optimizer replacing R lpSolve scripts"
```

---

## Task 8: Pipeline Orchestrator (CLI Entry Point)

**Files:**
- Create: `src/pipeline/run.py`
- Create: `tests/test_run.py`

- [ ] **Step 1: Write failing tests for the orchestrator**

```python
# tests/test_run.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.pipeline.run import phase_pre_deadline, phase_predict, phase_post_gw, phase_retrain


class TestPhasePreDeadline:
    def test_saves_xp_snapshot(self, tmp_path, sample_bootstrap_json):
        with patch("src.pipeline.run.fetch_bootstrap", return_value=sample_bootstrap_json), \
             patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.RESULTS_DIR", tmp_path / "results"):
            (tmp_path / "FPL" / "data" / "2025-26" / "gws").mkdir(parents=True)
            (tmp_path / "results").mkdir()

            next_gw = phase_pre_deadline()

        assert next_gw == 31
        xp_path = tmp_path / "FPL" / "data" / "2025-26" / "gws" / "xP31.csv"
        assert xp_path.exists()

    def test_saves_bootstrap_snapshot(self, tmp_path, sample_bootstrap_json):
        with patch("src.pipeline.run.fetch_bootstrap", return_value=sample_bootstrap_json), \
             patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.RESULTS_DIR", tmp_path / "results"):
            (tmp_path / "FPL" / "data" / "2025-26" / "gws").mkdir(parents=True)
            (tmp_path / "results").mkdir()

            phase_pre_deadline()

        snapshot_path = tmp_path / "results" / "snapshots" / "bootstrap_gw31.json"
        assert snapshot_path.exists()
        data = json.loads(snapshot_path.read_text())
        assert "elements" in data


class TestPhasePostGw:
    def test_saves_live_gw_csv(self, tmp_path, sample_bootstrap_json, sample_player_history_json):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_player_history_json

        gw_dir = tmp_path / "FPL" / "data" / "2025-26" / "gws"
        gw_dir.mkdir(parents=True)

        with patch("src.pipeline.run.fetch_bootstrap", return_value=sample_bootstrap_json), \
             patch("src.pipeline.run.fetch_fixtures", return_value=[]), \
             patch("src.pipeline.run.fetch_live_gw_data") as mock_fetch_live, \
             patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.RESULTS_DIR", tmp_path / "results"):
            import pandas as pd
            mock_fetch_live.return_value = pd.DataFrame({
                "name": ["Saka"], "element": [3], "GW": [30],
                "total_points": [8], "position": ["MID"], "team": ["Arsenal"],
            })
            (tmp_path / "results").mkdir()

            phase_post_gw()

        live_path = gw_dir / "gw30_live.csv"
        assert live_path.exists()


class TestPhasePredict:
    def test_writes_output_csvs(self, tmp_path, sample_bootstrap_json):
        import pandas as pd
        import numpy as np

        # Create minimal vaastav data
        gw_dir = tmp_path / "FPL" / "data" / "2025-26" / "gws"
        gw_dir.mkdir(parents=True)
        rows = []
        for gw in range(1, 10):
            rows.append({
                "name": "Saka", "position": "MID", "team": "Arsenal",
                "element": 3, "total_points": np.random.randint(2, 12),
                "minutes": 90, "goals_scored": 0, "assists": 0,
                "clean_sheets": 0, "ict_index": 10.0, "influence": 30.0,
                "creativity": 25.0, "threat": 40.0, "bps": 20, "bonus": 1,
                "value": 105, "transfers_in": 5000, "transfers_out": 1000,
                "selected": 3000000, "was_home": True, "opponent_team": 10,
                "fixture": gw, "round": gw, "GW": gw,
            })
        pd.DataFrame(rows).to_csv(gw_dir / "merged_gw.csv", index=False)

        # Save bootstrap snapshot for availability filtering
        results_dir = tmp_path / "results"
        snapshot_dir = results_dir / "snapshots"
        snapshot_dir.mkdir(parents=True)
        with open(snapshot_dir / "bootstrap_gw10.json", "w") as f:
            json.dump(sample_bootstrap_json, f)

        # Mock model to not exist → falls back to xP=0
        with patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.RESULTS_DIR", results_dir), \
             patch("src.pipeline.run.ACTIVE_MODEL", tmp_path / "nonexistent.sav"), \
             patch("src.pipeline.run.CURRENT_SEASON", "2025-26"):
            # Will warn about missing model, use fallback
            result = phase_predict(target_gw=10)

        assert (results_dir / "xi_gw10.csv").exists()
        assert (results_dir / "squad_gw10.csv").exists()


class TestPhaseRetrain:
    def test_saves_new_model(self, tmp_path):
        import pandas as pd
        import numpy as np

        # Create minimal vaastav data with enough rows
        gw_dir = tmp_path / "FPL" / "data" / "2025-26" / "gws"
        gw_dir.mkdir(parents=True)
        rows = []
        for player in range(1, 6):
            for gw in range(1, 20):
                rows.append({
                    "name": f"Player{player}", "position": "MID", "team": "Arsenal",
                    "element": player, "total_points": np.random.randint(0, 15),
                    "minutes": 90, "goals_scored": 0, "assists": 0,
                    "clean_sheets": 0, "ict_index": 10.0, "influence": 30.0,
                    "creativity": 25.0, "threat": 40.0, "bps": 20, "bonus": 1,
                    "value": 100, "transfers_in": 5000, "transfers_out": 1000,
                    "selected": 3000000, "was_home": True, "opponent_team": 10,
                    "fixture": gw, "round": gw, "GW": gw, "season": "2025-26",
                })
        pd.DataFrame(rows).to_csv(gw_dir / "merged_gw.csv", index=False)

        models_dir = tmp_path / "models"
        with patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.MODELS_DIR", models_dir), \
             patch("src.pipeline.run.ACTIVE_MODEL", models_dir / "rf_model.sav"), \
             patch("src.pipeline.run.CURRENT_SEASON", "2025-26"):
            phase_retrain(target_gw=32)

        assert (models_dir / "rf_model_gw32.sav").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the orchestrator**

```python
# src/pipeline/run.py
"""CLI entry point for the FPL weekly pipeline.

Usage:
    python -m src.pipeline.run pre-deadline   # Phase 1: fetch data + capture xP
    python -m src.pipeline.run predict        # Phase 2: generate predictions + optimize
    python -m src.pipeline.run post-gw        # Phase 3: collect results + live data patch
    python -m src.pipeline.run retrain        # Phase 4: retrain model (manual)
    python -m src.pipeline.run full           # Run phases 1-2 (for pre-deadline workflow)
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    VAASTAV_DIR, RESULTS_DIR, MODELS_DIR, CURRENT_SEASON,
    ACTIVE_MODEL, BOOTSTRAP_MAX_AGE_HOURS,
)
from src.pipeline.fetch import (
    fetch_bootstrap, get_current_gw, get_next_deadline,
    extract_xp_snapshot, fetch_fixtures, fetch_live_gw_data,
)
from src.pipeline.prepare import build_merged_dataset
from src.pipeline.features import engineer_features
from src.pipeline.predict import predict_next_gw, get_feature_columns
from src.pipeline.availability import filter_availability
from src.pipeline.optimize import optimize_team

logger = logging.getLogger(__name__)


def _load_cached_bootstrap(target_gw: int | None = None) -> dict | None:
    """Try to load a recent cached bootstrap snapshot."""
    snapshot_dir = RESULTS_DIR / "snapshots"
    if not snapshot_dir.exists():
        return None

    if target_gw:
        path = snapshot_dir / f"bootstrap_gw{target_gw}.json"
        if path.exists():
            return json.loads(path.read_text())

    # Find most recent snapshot
    snapshots = sorted(snapshot_dir.glob("bootstrap_gw*.json"), reverse=True)
    if not snapshots:
        return None

    path = snapshots[0]
    age_hours = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600
    if age_hours > BOOTSTRAP_MAX_AGE_HOURS:
        logger.warning(f"Cached bootstrap is {age_hours:.0f}h old (>{BOOTSTRAP_MAX_AGE_HOURS}h), skipping")
        return None

    return json.loads(path.read_text())


def phase_pre_deadline():
    """Phase 1: Fetch bootstrap data and capture xP before deadline."""
    print("[pre-deadline] Fetching FPL API bootstrap...")
    try:
        bootstrap = fetch_bootstrap()
    except Exception as e:
        logger.error(f"API fetch failed: {e}")
        bootstrap = _load_cached_bootstrap()
        if bootstrap is None:
            print("[pre-deadline] ERROR: API unreachable and no valid cached bootstrap. Aborting.")
            return None
        print("[pre-deadline] Using cached bootstrap snapshot")

    gw = get_current_gw(bootstrap)
    next_gw, deadline = get_next_deadline(bootstrap)
    print(f"[pre-deadline] Current GW: {gw}, Next deadline: GW{next_gw} at {deadline}")

    # Capture xP snapshot
    xp = extract_xp_snapshot(bootstrap)
    xp_path = VAASTAV_DIR / "data" / CURRENT_SEASON / "gws" / f"xP{next_gw}.csv"
    xp_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(xp.items()), columns=["id", "xP"]).to_csv(xp_path, index=False)
    print(f"[pre-deadline] Saved xP snapshot for GW{next_gw} ({len(xp)} players)")

    # Save bootstrap for reference
    snapshot_dir = RESULTS_DIR / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with open(snapshot_dir / f"bootstrap_gw{next_gw}.json", "w") as f:
        json.dump(bootstrap, f)
    print("[pre-deadline] Saved bootstrap snapshot")

    return next_gw


def phase_predict(target_gw: int | None = None):
    """Phase 2: Build features, predict, filter availability, optimize."""
    print("[predict] Building merged dataset...")
    merged = build_merged_dataset()
    print(f"[predict] Dataset: {len(merged)} rows, {len(merged.columns)} columns")

    print("[predict] Engineering features...")
    features = engineer_features(merged)
    print(f"[predict] Features: {len(features)} rows after NaN drop")

    # Get latest row per player for prediction
    latest = features.sort_values(["element", "GW"]).groupby("element").last().reset_index()

    # Ensure now_cost column exists (vaastav uses 'value', FPL API uses 'now_cost')
    if "now_cost" not in latest.columns:
        latest["now_cost"] = latest.get("value", pd.Series(50, index=latest.index))

    # Load bootstrap for cost override and availability filtering
    bootstrap = None
    if target_gw:
        bootstrap = _load_cached_bootstrap(target_gw)
        if bootstrap:
            cost_map = {e["id"]: e["now_cost"] for e in bootstrap["elements"]}
            latest["now_cost"] = latest["element"].map(cost_map).fillna(latest["now_cost"])

    print("[predict] Generating predictions...")
    model_path = ACTIVE_MODEL
    if not model_path.exists():
        print(f"[predict] WARNING: Model not found at {model_path}. Using xP from API.")
        if target_gw:
            xp_path = VAASTAV_DIR / "data" / CURRENT_SEASON / "gws" / f"xP{target_gw}.csv"
            if xp_path.exists():
                xp_df = pd.read_csv(xp_path)
                latest = latest.merge(
                    xp_df.rename(columns={"id": "element"}),
                    on="element", how="left", suffixes=("_feat", ""),
                )
        predictions = latest[["element", "name", "position", "team"]].copy()
        predictions["xP"] = latest.get("xP", 0)
        predictions["now_cost"] = latest["now_cost"]
    else:
        predictions = predict_next_gw(latest, model_path)

    # Apply availability filtering
    if bootstrap:
        print("[predict] Filtering by player availability...")
        before_count = len(predictions)
        predictions = filter_availability(predictions, bootstrap)
        excluded = before_count - len(predictions)
        if excluded > 0:
            print(f"[predict] Excluded {excluded} unavailable players")
    else:
        print("[predict] Skipping availability filter (no bootstrap data)")

    print("[predict] Optimizing team selection...")
    result = optimize_team(predictions)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gw_label = f"gw{target_gw}" if target_gw else "latest"
    result["xi"].to_csv(RESULTS_DIR / f"xi_{gw_label}.csv", index=False)
    result["squad"].to_csv(RESULTS_DIR / f"squad_{gw_label}.csv", index=False)

    print(f"\n{'='*50}")
    print(f"OPTIMAL XI for GW{target_gw or '?'}:")
    print(f"{'='*50}")
    xi = result["xi"].sort_values("position")
    for _, p in xi.iterrows():
        cap = " (C)" if p["element"] == result["captain"]["element"] else ""
        vc = " (VC)" if p["element"] == result["vice_captain"]["element"] else ""
        print(f"  {p['position']:3s} | {p['name']:20s} | {p['team']:15s} | xP: {p['xP']:.1f}{cap}{vc}")
    print(f"\nTotal xP (with captain): {result['total_xp']:.1f}")
    print(f"Budget used: {result['squad']['now_cost'].sum() / 10:.1f}M")

    return result


def phase_post_gw():
    """Phase 3: Collect actual results and save live GW data."""
    print("[post-gw] Fetching updated bootstrap...")
    try:
        bootstrap = fetch_bootstrap()
    except Exception as e:
        logger.error(f"API fetch failed during post-gw: {e}")
        print("[post-gw] ERROR: API unreachable. Skipping live data collection.")
        return

    gw = get_current_gw(bootstrap)
    print(f"[post-gw] Current GW: {gw}")

    # Fetch fixtures for actual scores
    print("[post-gw] Fetching fixtures...")
    fixtures = fetch_fixtures()
    finished = [f for f in fixtures if f.get("finished") and f.get("event") == gw]
    print(f"[post-gw] {len(finished)} finished fixtures in GW{gw}")

    # Collect live player data for this GW
    print(f"[post-gw] Fetching player histories for GW{gw}...")
    live_df = fetch_live_gw_data(target_gw=gw, bootstrap_data=bootstrap)

    if not live_df.empty:
        gw_dir = VAASTAV_DIR / "data" / CURRENT_SEASON / "gws"
        gw_dir.mkdir(parents=True, exist_ok=True)
        live_path = gw_dir / f"gw{gw}_live.csv"
        live_df.to_csv(live_path, index=False)
        print(f"[post-gw] Saved {len(live_df)} player rows to {live_path}")
    else:
        print("[post-gw] No player data collected (GW may not be finished)")

    print("[post-gw] Done. Run 'predict' to update features with new data.")


def phase_retrain(target_gw: int | None = None):
    """Phase 4: Retrain RF model on full dataset (manual trigger)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    import joblib

    print("[retrain] Building full feature-engineered dataset...")
    merged = build_merged_dataset()
    features = engineer_features(merged)
    print(f"[retrain] Training data: {len(features)} rows")

    feature_cols = get_feature_columns()
    X = features[feature_cols].fillna(0)
    y = features["total_points"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[retrain] Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"[retrain] New model — MAE: {mae:.2f}, R2: {r2:.3f}")

    # Compare with existing model if available
    if ACTIVE_MODEL.exists():
        old_model = joblib.load(ACTIVE_MODEL)
        old_pred = old_model.predict(X_test)
        old_mae = mean_absolute_error(y_test, old_pred)
        old_r2 = r2_score(y_test, old_pred)
        print(f"[retrain] Old model — MAE: {old_mae:.2f}, R2: {old_r2:.3f}")
        if mae < old_mae:
            print("[retrain] New model is BETTER (lower MAE)")
        else:
            print("[retrain] New model is WORSE (higher MAE) — consider keeping old model")

    # Save with GW label (or timestamp fallback)
    label = f"gw{target_gw}" if target_gw else datetime.now().strftime("%Y%m%d_%H%M%S")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    new_path = MODELS_DIR / f"rf_model_{label}.sav"
    joblib.dump(model, new_path)
    print(f"[retrain] Saved new model to {new_path}")
    print(f"[retrain] To promote: update ACTIVE_MODEL in src/config.py to point to {new_path.name}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="FPL Weekly Pipeline")
    parser.add_argument("phase", choices=["pre-deadline", "predict", "post-gw", "retrain", "full"],
                        help="Pipeline phase to run")
    parser.add_argument("--gw", type=int, help="Target gameweek (optional)")
    args = parser.parse_args()

    if args.phase == "pre-deadline":
        phase_pre_deadline()
    elif args.phase == "predict":
        phase_predict(args.gw)
    elif args.phase == "post-gw":
        phase_post_gw()
    elif args.phase == "retrain":
        phase_retrain(args.gw)
    elif args.phase == "full":
        gw = phase_pre_deadline()
        if gw:
            phase_predict(gw)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/run.py tests/test_run.py
git commit -m "feat: add pipeline orchestrator with 4 phases and availability filtering"
```

---

## Task 9: Weekly Run Script

**Files:**
- Create: `scripts/weekly_run.sh`

- [ ] **Step 1: Write the cron wrapper**

```bash
#!/bin/bash
# scripts/weekly_run.sh — Cron-friendly FPL pipeline wrapper
#
# Schedule in crontab (adjust times per GW calendar):
#   GW32: 0 17 10 4 * /path/to/scripts/weekly_run.sh pre-deadline
#   GW32: 0 22 10 4 * /path/to/scripts/weekly_run.sh predict
#
# Or run manually:
#   ./scripts/weekly_run.sh pre-deadline
#   ./scripts/weekly_run.sh predict
#   ./scripts/weekly_run.sh post-gw
#   ./scripts/weekly_run.sh retrain

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

PHASE="${1:-full}"
LOGDIR="$PROJECT_DIR/logs"
mkdir -p "$LOGDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOGDIR/${PHASE}_${TIMESTAMP}.log"

echo "[$(date)] Starting phase: $PHASE" | tee "$LOGFILE"
python -m src.pipeline.run "$PHASE" "${@:2}" 2>&1 | tee -a "$LOGFILE"
echo "[$(date)] Completed phase: $PHASE" | tee -a "$LOGFILE"
```

- [ ] **Step 2: Make executable and test**

Run: `chmod +x scripts/weekly_run.sh && bash scripts/weekly_run.sh predict --gw 30`

> **Windows note:** This script runs under Git Bash or WSL. For native Windows scheduling, use Task Scheduler instead of cron. Alternatively, run directly: `python -m src.pipeline.run predict --gw 30`

- [ ] **Step 3: Commit**

```bash
git add scripts/weekly_run.sh
git commit -m "feat: add cron-friendly weekly run script with logging"
```

---

## Task 10: Integration Test — End-to-End with Vaastav Data

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration test: runs the full pipeline on real vaastav data (2024-25 season)."""
import pytest
import pandas as pd
from pathlib import Path
from src.config import VAASTAV_DIR
from src.pipeline.prepare import build_merged_dataset
from src.pipeline.features import engineer_features
from src.pipeline.optimize import optimize_team


@pytest.mark.skipif(
    not (VAASTAV_DIR / "data" / "2024-25" / "gws" / "merged_gw.csv").exists(),
    reason="Vaastav dataset not cloned"
)
class TestIntegration:
    def test_full_pipeline_2024_25(self):
        """Run prepare -> features -> optimize on real 2024-25 data."""
        # Prepare
        merged = build_merged_dataset(seasons=["2024-25"])
        assert len(merged) > 5000

        # Features
        features = engineer_features(merged)
        assert len(features) > 1000
        assert "total_points_roll_4" in features.columns

        # Get latest row per player
        latest = features.sort_values(["element", "GW"]).groupby("element").last().reset_index()

        # Build optimizer input (use actual total_points as proxy for xP)
        optimizer_input = latest[["element", "name", "position", "team"]].copy()
        optimizer_input["xP"] = latest["total_points_roll_4"].fillna(2.0)
        optimizer_input["now_cost"] = latest["value"].fillna(50)

        # Optimize
        result = optimize_team(optimizer_input)

        assert len(result["squad"]) == 15
        assert len(result["xi"]) == 11
        assert result["total_xp"] > 0

        # Position sanity
        xi_pos = result["xi"]["position"].value_counts()
        assert xi_pos["GK"] == 1
        assert 3 <= xi_pos.get("DEF", 0) <= 5
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v -s`
Expected: PASS (may take 10-15 seconds)

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test on real vaastav data"
```

---

## Task 11: GW32 Live Dry Run

**Files:** None new — this is a manual verification step.

- [ ] **Step 1: Update vaastav data**

```bash
cd data/Fantasy-Premier-League && git pull && cd ../..
```

- [ ] **Step 2: Run pre-deadline phase**

```bash
python -m src.pipeline.run pre-deadline
```

Expected: Captures xP snapshot for next GW, saves bootstrap JSON.

- [ ] **Step 3: Run predict phase**

```bash
python -m src.pipeline.run predict
```

Expected: Prints optimal XI with xP values, shows excluded/scaled players. Saves `xi_gwNN.csv` and `squad_gwNN.csv` to `results/`.

- [ ] **Step 4: Verify outputs**

```bash
cat results/xi_gw*.csv | head -15
cat results/squad_gw*.csv | head -20
```

Expected: 11-row XI CSV, 15-row squad CSV, reasonable xP values and team composition.

- [ ] **Step 5: Commit results snapshot**

```bash
git add results/xi_gw*.csv results/squad_gw*.csv results/snapshots/
git commit -m "data: GW32 dry run — first live pipeline execution"
```

---

## Verification

### Run all tests
```bash
pytest tests/ -v --tb=short
```
Expected: All tests pass (unit + integration).

### Manual verification
1. `python -m src.pipeline.run predict --gw 30` completes without errors
2. `results/xi_gw30.csv` contains 11 players with valid positions
3. `results/squad_gw30.csv` contains 15 players within budget
4. No hardcoded absolute paths in any new file

### Pre-GW32 checklist (before Apr 10, 2026 17:30 UTC)
1. Run `python -m src.pipeline.run pre-deadline` to capture xP
2. Run `python -m src.pipeline.run predict` to generate team
3. Review the XI output — check excluded/scaled players make sense
4. Make transfers in FPL app
5. After GW completes: `python -m src.pipeline.run post-gw` to collect results

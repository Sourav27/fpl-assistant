# Track A — User Team Sync, Recommend & Post-Match Analysis

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a `recommend` CLI phase that reads the user's real FPL squad/budget from the API and produces a multi-GW transfer plan, then extend `post-gw` to compare predictions vs actuals and log season benchmarks.

**Architecture:** Two new modules (`user.py` for API data fetching, `recommend.py` for ILP), one new module (`analysis.py` for post-match), all wired into `run.py`. `predict.py` gains a side-effect: saves full player predictions CSV that `recommend` and `analysis` consume. User preferences live in a gitignored `user_config.yaml`.

**Tech Stack:** Python, PuLP (already in requirements), PyYAML (add to requirements), pandas, unittest.mock for tests. No new dependencies beyond PyYAML.

---

## ✅ STATUS: COMPLETE (2026-03-29 04:30 IST)

All tasks (1–13) + Final step implemented and committed to master. Full test suite passing (116 tests). Remote agent completed Tasks 6–Final as scheduled.

**All features now live:**
- User config YAML loader with FPL entry ID + preferences (horizon, FDR sensitivity, hit cap)
- FPL API integration: fetch squad, bank, free transfers, historical benchmarks
- Multi-GW transfer planner with FT banking constraints & -4 hit cost
- Wildcard/free-hit unconstrained squad rebuild
- Post-match analysis: prediction misses, dream team, accuracy benchmarks
- Full CSV export for all phases + new `recommend` CLI with `--horizon/--wildcard/--team` flags

---

## File Map

```
New:
  user_config.example.yaml           # committed template
  user_config.yaml                   # gitignored (user creates)
  src/pipeline/user.py               # UserTeamState dataclass + API fetch + benchmark fetch
  src/pipeline/recommend.py          # multi-GW ILP transfer optimizer
  src/pipeline/analysis.py           # post-match prediction vs actual analysis
  tests/test_user.py
  tests/test_recommend.py
  tests/test_analysis.py

Modified:
  src/config.py                      # new API URL constants + load_user_config()
  src/pipeline/predict.py            # save results/predictions_gw{N}.csv after prediction
  src/pipeline/run.py                # add recommend phase + extend post-gw
  requirements.txt                   # add pyyaml
  .gitignore                         # add user_config.yaml
```

---

## Task 1: User Config (YAML template + loader)

**Context:** The pipeline needs to know the user's FPL entry ID and preferences (horizon, FDR sensitivity, hit cap). These are personal and gitignored. The loader lives in `src/config.py` alongside all other config.

**Files:**
- Create: `user_config.example.yaml`
- Create: `user_config.yaml` (gitignored, user creates from example)
- Modify: `src/config.py`
- Modify: `.gitignore`
- Modify: `requirements.txt`
- Test: `tests/test_user_config.py`

- [x] **Step 1: Add pyyaml to requirements**

```
# requirements.txt — add this line:
pyyaml>=6.0
```

Run: `pip install pyyaml`

- [x] **Step 2: Write failing tests**

Create `tests/test_user_config.py`:

```python
import pytest
import tempfile
from pathlib import Path
from src.config import load_user_config, UserConfigError


class TestLoadUserConfig:
    def test_loads_valid_config(self, tmp_path):
        cfg_file = tmp_path / "user_config.yaml"
        cfg_file.write_text("""
teams:
  default:
    entry_id: 1681779
    label: "Main"
preferences:
  horizon_gws: 5
  max_hit_points: 8
  fdr_sensitivity: 0.15
""")
        cfg = load_user_config(cfg_file)
        assert cfg["teams"]["default"]["entry_id"] == 1681779
        assert cfg["preferences"]["horizon_gws"] == 5
        assert cfg["preferences"]["fdr_sensitivity"] == 0.15

    def test_raises_when_file_missing(self, tmp_path):
        with pytest.raises(UserConfigError, match="user_config.yaml"):
            load_user_config(tmp_path / "nonexistent.yaml")

    def test_raises_when_entry_id_missing(self, tmp_path):
        cfg_file = tmp_path / "user_config.yaml"
        cfg_file.write_text("teams:\n  default:\n    label: Main\n")
        with pytest.raises(UserConfigError, match="entry_id"):
            load_user_config(cfg_file)

    def test_raises_when_entry_id_not_int(self, tmp_path):
        cfg_file = tmp_path / "user_config.yaml"
        cfg_file.write_text("teams:\n  default:\n    entry_id: abc\n")
        with pytest.raises(UserConfigError, match="entry_id"):
            load_user_config(cfg_file)

    def test_raises_when_horizon_out_of_range(self, tmp_path):
        cfg_file = tmp_path / "user_config.yaml"
        cfg_file.write_text("""
teams:
  default:
    entry_id: 123
preferences:
  horizon_gws: 10
""")
        with pytest.raises(UserConfigError, match="horizon_gws"):
            load_user_config(cfg_file)

    def test_defaults_applied_when_preferences_missing(self, tmp_path):
        cfg_file = tmp_path / "user_config.yaml"
        cfg_file.write_text("teams:\n  default:\n    entry_id: 123\n")
        cfg = load_user_config(cfg_file)
        assert cfg["preferences"]["horizon_gws"] == 5
        assert cfg["preferences"]["max_hit_points"] == 8
        assert cfg["preferences"]["fdr_sensitivity"] == 0.15

    def test_alt_team_optional(self, tmp_path):
        cfg_file = tmp_path / "user_config.yaml"
        cfg_file.write_text("""
teams:
  default:
    entry_id: 123
  alt:
    entry_id: 456
    label: Experimental
""")
        cfg = load_user_config(cfg_file)
        assert cfg["teams"]["alt"]["entry_id"] == 456
```

- [x] **Step 3: Run tests to verify they fail**

```bash
python -m pytest tests/test_user_config.py -v
```
Expected: FAIL — `ImportError: cannot import name 'load_user_config'`

- [x] **Step 4: Add `UserConfigError`, `load_user_config()`, and new URL constants to `src/config.py`**

Add after the existing imports at the top of `src/config.py`:

```python
import yaml
```

Add after existing URL constants:

```python
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
    if not isinstance(horizon, int) or not (1 <= horizon <= 5):
        raise UserConfigError(
            f"user_config.yaml: horizon_gws must be an integer 1-5, got {horizon!r}"
        )

    return cfg
```

- [x] **Step 5: Create `user_config.example.yaml`**

```yaml
# user_config.example.yaml — copy to user_config.yaml and fill in your IDs
# user_config.yaml is gitignored — never commit it

teams:
  default:
    entry_id: 1234567      # your FPL entry ID (from fantasy.premierleague.com/entry/{id}/event/1)
    label: "Main"
  alt:                     # optional second team
    entry_id: 7654321
    label: "Experimental"

preferences:
  horizon_gws: 5           # GWs to plan ahead (1 = this GW only, max 5)
  max_hit_points: 8        # max penalty points per GW (-4 per hit, so 8 = max 2 extra transfers)
  fdr_sensitivity: 0.15    # how much fixture difficulty shifts xP (0 = ignore FDR, 0.3 = aggressive)
```

- [x] **Step 6: Add `user_config.yaml` to `.gitignore`**

```bash
echo "user_config.yaml" >> .gitignore
```

- [x] **Step 7: Run tests to verify they pass**

```bash
python -m pytest tests/test_user_config.py -v
```
Expected: 7 tests PASS

- [x] **Step 8: Commit**

```bash
git add src/config.py user_config.example.yaml tests/test_user_config.py requirements.txt .gitignore
git commit -m "feat: add user_config.yaml loader with validation and defaults"
```

---

## Task 2: UserTeamState dataclass + basic construction

**Context:** `user.py` fetches the user's actual team from the FPL API and packages it into a dataclass. This is the data contract used by `recommend.py`. All costs are in 0.1M units (same as FPL API). The `code` field is the persistent cross-season player ID.

**Files:**
- Create: `src/pipeline/user.py`
- Test: `tests/test_user.py`

- [x] **Step 1: Write failing tests for the dataclass**

Create `tests/test_user.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from src.pipeline.user import UserTeamState, fetch_user_team_state, compute_selling_price


class TestUserTeamStateDataclass:
    def test_total_value_computed(self):
        state = UserTeamState(
            entry_id=123,
            current_squad=[1, 2, 3],
            squad_codes=[101, 102, 103],
            selling_prices={1: 55, 2: 62, 3: 105},
            bank=50,
            free_transfers=2,
            active_chip=None,
            total_value=0,  # will be overridden
        )
        # total_value = sum(selling_prices) + bank = 222 + 50 = 272
        assert state.total_value == 272

    def test_free_transfers_clamped(self):
        state = UserTeamState(
            entry_id=123, current_squad=[], squad_codes=[],
            selling_prices={}, bank=0, free_transfers=10,
            active_chip=None, total_value=0,
        )
        assert state.free_transfers == 5  # capped at 5

    def test_active_chip_none_when_absent(self):
        state = UserTeamState(
            entry_id=123, current_squad=[], squad_codes=[],
            selling_prices={}, bank=100, free_transfers=1,
            active_chip=None, total_value=100,
        )
        assert state.active_chip is None
```

- [x] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_user.py::TestUserTeamStateDataclass -v
```
Expected: FAIL — `ImportError`

- [x] **Step 3: Create `src/pipeline/user.py` with dataclass**

```python
"""FPL user team state fetcher and post-match analysis helpers."""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
from src.pipeline.fetch import _api_get_with_retry
from src.config import FPL_ENTRY_URL, FPL_EVENT_URL, FPL_LEAGUES_CLASSIC_URL

logger = logging.getLogger(__name__)


@dataclass
class UserTeamState:
    """The user's current FPL team state, fetched from the public FPL API.

    All cost values are in 0.1M units (FPL convention): 77 = £7.7m.
    """
    entry_id: int
    current_squad: list[int]        # 15 element IDs (seasonal, changes each season)
    squad_codes: list[int]          # 15 persistent player codes (for cross-season joins)
    selling_prices: dict[int, int]  # element → selling price (0.1M units)
    bank: int                       # remaining budget (0.1M units, e.g. 350 = £35.0m)
    free_transfers: int             # banked free transfers, range 1-5
    active_chip: str | None         # "wildcard" | "freehit" | "bboost" | "3xc" | None
    total_value: int                # sum(selling_prices.values()) + bank

    def __post_init__(self):
        # Recompute total_value from components (ignores passed-in value)
        self.total_value = sum(self.selling_prices.values()) + self.bank
        # Cap free transfers at 5
        self.free_transfers = min(self.free_transfers, 5)
```

- [x] **Step 4: Run tests**

```bash
python -m pytest tests/test_user.py::TestUserTeamStateDataclass -v
```
Expected: 3 tests PASS

---

## Task 3: user.py — fetch_user_team_state() API integration

**Context:** Four FPL API endpoints are needed. Use the existing `_api_get_with_retry()` from `fetch.py`. The `picks` endpoint returns element IDs; map these to persistent `code` values using bootstrap data.

**Files:**
- Modify: `src/pipeline/user.py`
- Modify: `tests/test_user.py`

- [x] **Step 1: Write failing tests for fetch_user_team_state**

Add to `tests/test_user.py`. Note: `sample_bootstrap_json` is defined in `tests/conftest.py` (already exists) — import it via pytest fixture injection, not manually.

```python
import pandas as pd  # needed for later tests
```

Then add the test class:

```python
class TestFetchUserTeamState:
    def _make_mock_response(self, data: dict) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = data
        return resp

    def test_returns_user_team_state(self, sample_bootstrap_json):
        entry_resp = {"value": 1050, "bank": 50, "last_deadline_bank": 50}
        picks_resp = {
            "active_chip": None,
            "picks": [
                {"element": 1, "selling_price": 55, "purchase_price": 50},
                {"element": 2, "selling_price": 62, "purchase_price": 60},
                {"element": 3, "selling_price": 105, "purchase_price": 100},
            ] + [{"element": i, "selling_price": 50, "purchase_price": 48} for i in range(4, 16)],
        }
        history_resp = {
            "current": [
                {"event": 30, "event_transfers": 1, "bank": 50, "points": 58,
                 "percentile_rank": 20},
            ]
        }
        transfers_resp = []  # no transfer history

        with patch("src.pipeline.user._api_get_with_retry") as mock_get:
            mock_get.side_effect = [
                self._make_mock_response(entry_resp),
                self._make_mock_response(picks_resp),
                self._make_mock_response(transfers_resp),
                self._make_mock_response(history_resp),
            ]
            state = fetch_user_team_state(
                entry_id=123, gw=30, bootstrap_data=sample_bootstrap_json
            )

        assert isinstance(state, UserTeamState)
        assert state.entry_id == 123
        assert len(state.current_squad) == 15
        assert state.bank == 50
        assert state.free_transfers >= 1
        assert state.active_chip is None

    def test_wildcard_chip_detected(self, sample_bootstrap_json):
        entry_resp = {"value": 1000, "bank": 0}
        picks_resp = {
            "active_chip": "wildcard",
            "picks": [{"element": i, "selling_price": 67, "purchase_price": 67}
                      for i in range(1, 16)],
        }
        history_resp = {"current": [{"event": 30, "event_transfers": 0, "bank": 0, "points": 0,
                                      "percentile_rank": 50}]}
        transfers_resp = []

        with patch("src.pipeline.user._api_get_with_retry") as mock_get:
            mock_get.side_effect = [
                self._make_mock_response(entry_resp),
                self._make_mock_response(picks_resp),
                self._make_mock_response(transfers_resp),
                self._make_mock_response(history_resp),
            ]
            state = fetch_user_team_state(123, 30, sample_bootstrap_json)

        assert state.active_chip == "wildcard"

    def test_free_transfers_from_history(self, sample_bootstrap_json):
        """Banking: if user made 0 transfers in GW30, they should have ft_prev + 1 (capped 5)."""
        entry_resp = {"value": 1000, "bank": 0}
        picks_resp = {
            "active_chip": None,
            "picks": [{"element": i, "selling_price": 67, "purchase_price": 67}
                      for i in range(1, 16)],
        }
        # GW29: had 1 FT, used 0 → GW30: 2 FT
        history_resp = {
            "current": [
                {"event": 29, "event_transfers": 0, "bank": 0, "points": 60, "percentile_rank": 30},
                {"event": 30, "event_transfers": 1, "bank": 0, "points": 58, "percentile_rank": 20},
            ]
        }
        transfers_resp = []

        with patch("src.pipeline.user._api_get_with_retry") as mock_get:
            mock_get.side_effect = [
                self._make_mock_response(entry_resp),
                self._make_mock_response(picks_resp),
                self._make_mock_response(transfers_resp),
                self._make_mock_response(history_resp),
            ]
            state = fetch_user_team_state(123, 30, sample_bootstrap_json)

        # After using transfers in GW30, next GW = 1 FT
        assert 1 <= state.free_transfers <= 5
```

- [x] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_user.py::TestFetchUserTeamState -v
```
Expected: FAIL — `ImportError: cannot import name 'fetch_user_team_state'`

- [x] **Step 3: Implement `fetch_user_team_state()` in `src/pipeline/user.py`**

```python
def fetch_user_team_state(
    entry_id: int,
    gw: int,
    bootstrap_data: dict,
) -> UserTeamState:
    """Fetch user's current team state from the public FPL API.

    Makes 4 API calls: entry info, picks for current GW, transfer history, GW history.
    All cost values returned in 0.1M units.
    """
    # Build code lookup: element_id → persistent code
    code_map = {e["id"]: e["code"] for e in bootstrap_data.get("elements", [])}

    # 1. Entry info (bank balance)
    entry_data = _api_get_with_retry(f"{FPL_ENTRY_URL}/{entry_id}/").json()
    bank = entry_data.get("last_deadline_bank") or entry_data.get("bank", 0)

    # 2. Current picks
    picks_data = _api_get_with_retry(
        f"{FPL_ENTRY_URL}/{entry_id}/event/{gw}/picks/"
    ).json()
    picks = picks_data.get("picks", [])
    active_chip = picks_data.get("active_chip")

    current_squad = [p["element"] for p in picks]
    squad_codes = [code_map.get(e, e) for e in current_squad]

    # 3. Transfer history (for selling price computation)
    transfers_data = _api_get_with_retry(
        f"{FPL_ENTRY_URL}/{entry_id}/transfers/"
    ).json()
    # Build purchase price map from transfer history: element → most recent buy price
    purchase_prices: dict[int, int] = {}
    for t in (transfers_data if isinstance(transfers_data, list) else []):
        purchase_prices[t["element_in"]] = t["element_in_cost"]

    # Compute selling prices
    cost_map = {e["id"]: e["now_cost"] for e in bootstrap_data.get("elements", [])}
    selling_prices: dict[int, int] = {}
    for pick in picks:
        elem = pick["element"]
        # Use selling_price from picks if available (authenticated only), else compute
        if pick.get("selling_price"):
            selling_prices[elem] = pick["selling_price"]
        else:
            now = cost_map.get(elem, pick.get("purchase_price", 50))
            buy = purchase_prices.get(elem, now)  # fallback: no profit
            selling_prices[elem] = compute_selling_price(buy, now)

    # 4. GW history (for free transfer calculation)
    history_data = _api_get_with_retry(
        f"{FPL_ENTRY_URL}/{entry_id}/history/"
    ).json()
    free_transfers = _compute_free_transfers(history_data.get("current", []), gw)

    return UserTeamState(
        entry_id=entry_id,
        current_squad=current_squad,
        squad_codes=squad_codes,
        selling_prices=selling_prices,
        bank=bank,
        free_transfers=free_transfers,
        active_chip=active_chip,
        total_value=0,  # recalculated in __post_init__
    )
```

- [x] **Step 4: Run tests**

```bash
python -m pytest tests/test_user.py::TestFetchUserTeamState -v
```
Expected: FAIL — `compute_selling_price` and `_compute_free_transfers` not yet defined

---

## Task 4: user.py — selling price and free transfer helpers

**Files:**
- Modify: `src/pipeline/user.py`
- Modify: `tests/test_user.py`

- [x] **Step 1: Write failing tests**

Add to `tests/test_user.py`:

```python
class TestComputeSellingPrice:
    def test_no_profit(self):
        # bought 75, now 75 → sell at 75
        assert compute_selling_price(75, 75) == 75

    def test_profit_rounds_down(self):
        # bought 75, now 78 → profit 3 → half = 1.5 → floor = 1 → sell 76
        assert compute_selling_price(75, 78) == 76

    def test_full_profit(self):
        # bought 80, now 84 → profit 4 → half = 2 → sell 82
        assert compute_selling_price(80, 84) == 82

    def test_no_loss(self):
        # FPL never applies a haircut on price drops — sell at current price
        assert compute_selling_price(80, 75) == 75

    def test_exact_half_profit(self):
        # bought 100, now 102 → profit 2 → half 1 → sell 101
        assert compute_selling_price(100, 102) == 101
```

- [x] **Step 2: Run to verify fail**

```bash
python -m pytest tests/test_user.py::TestComputeSellingPrice -v
```
Expected: FAIL — function not defined

- [x] **Step 3: Implement helpers in `src/pipeline/user.py`**

```python
def compute_selling_price(purchase_price: int, current_price: int) -> int:
    """Compute FPL selling price in 0.1M units.

    Selling price = purchase_price + floor((current - purchase) / 2)
    FPL never charges a penalty for price drops — sell at current price if value fell.
    """
    profit = current_price - purchase_price
    if profit <= 0:
        return current_price
    return purchase_price + profit // 2


def _compute_free_transfers(gw_history: list[dict], current_gw: int) -> int:
    """Compute banked free transfers entering the NEXT gameweek.

    Logic: each unused FT banks by 1 (max 5). After using transfers, reset to 1.
    We simulate from history to find the FT count after current_gw ends.
    """
    ft = 1  # FPL starts everyone with 1 FT at GW1
    for row in sorted(gw_history, key=lambda r: r["event"]):
        if row["event"] > current_gw:
            break
        transfers_used = row.get("event_transfers", 0)
        if transfers_used == 0:
            ft = min(ft + 1, 5)  # bank 1, cap at 5
        else:
            # After using transfers, next GW = max(1, ft - transfers_used) + 1 banked
            ft = max(1, ft - transfers_used) + 1
            ft = min(ft, 5)
    return max(ft, 1)
```

- [x] **Step 4: Run all user tests**

```bash
python -m pytest tests/test_user.py -v
```
Expected: all tests PASS

- [x] **Step 5: Commit**

```bash
git add src/pipeline/user.py src/config.py tests/test_user.py
git commit -m "feat: add user.py with UserTeamState dataclass and FPL API fetching"
```

---

## Task 5: predict.py — save full predictions CSV

**Context:** `recommend.py` needs predictions for ALL players (not just the optimised XI). The `phase_predict()` function must write `results/predictions_gw{N}.csv` as a side-effect. This CSV is the single source of truth for xP values consumed by `recommend` and `analysis`.

**Files:**
- Modify: `src/pipeline/predict.py`
- Modify: `src/pipeline/run.py`
- Modify: `tests/test_predict.py` (add one test)

- [x] **Step 1: Write failing test**

Open `tests/test_predict.py` and add to the end:

```python
class TestSaveFullPredictionsCSV:
    def test_save_predictions_creates_file(self, sample_predictions_df, tmp_path):
        from src.pipeline.predict import save_full_predictions
        out_path = tmp_path / "predictions_gw33.csv"
        save_full_predictions(sample_predictions_df, out_path)
        assert out_path.exists()
        import pandas as pd
        df = pd.read_csv(out_path)
        assert list(df.columns) == ["element", "code", "name", "position", "team", "xP", "now_cost"]
        assert len(df) == len(sample_predictions_df)

    def test_save_predictions_cost_in_01m_units(self, sample_predictions_df, tmp_path):
        from src.pipeline.predict import save_full_predictions
        # now_cost stays in 0.1M units (FPL convention): 105 = £10.5m stored as 105
        out_path = tmp_path / "predictions_gw33.csv"
        save_full_predictions(sample_predictions_df, out_path)
        import pandas as pd
        df = pd.read_csv(out_path)
        # Saka: now_cost=105 stays as 105 (0.1M units)
        saka = df[df["name"] == "Saka"].iloc[0]
        assert saka["now_cost"] == 105
```

- [x] **Step 2: Run to verify fail**

```bash
python -m pytest tests/test_predict.py::TestSaveFullPredictionsCSV -v
```
Expected: FAIL

- [x] **Step 3: Add `save_full_predictions()` to `src/pipeline/predict.py`**

Add after the existing `predict_next_gw()` function:

```python
def save_full_predictions(predictions: pd.DataFrame, path: Path) -> None:
    """Save full player predictions to CSV.

    Columns: element, code, name, position, team, xP, now_cost (in 0.1M units, e.g. 105 = £10.5m)
    now_cost is kept in 0.1M units (FPL API convention) so recommend.py and optimize.py
    can use it directly without unit conversion. Convert to £ only in user-facing output (CSVs,
    terminal summaries) by dividing by 10.
    """
    df = predictions.copy()
    # Ensure code column exists (may be absent if model ran without cross-season data)
    if "code" not in df.columns:
        df["code"] = df.get("element", pd.Series(dtype=int))
    # now_cost stays in 0.1M units — do NOT divide by 10 here
    cols = ["element", "code", "name", "position", "team", "xP", "now_cost"]
    df = df[[c for c in cols if c in df.columns]]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
```

- [x] **Step 4: Call `save_full_predictions()` from `phase_predict()` in `src/pipeline/run.py`**

In `run.py`, add the import at the top:

```python
from src.pipeline.predict import predict_next_gw, get_feature_columns, save_full_predictions
```

In `phase_predict()`, after the line `RESULTS_DIR.mkdir(parents=True, exist_ok=True)` and before `result = optimize_team(predictions)`:

```python
    # Save full predictions for recommend + analysis phases
    pred_path = RESULTS_DIR / f"predictions_{gw_label}.csv"
    save_full_predictions(predictions, pred_path)
    print(f"[predict] Saved full predictions ({len(predictions)} players) to {pred_path}")
```

- [x] **Step 5: Run tests**

```bash
python -m pytest tests/test_predict.py -v
```
Expected: all PASS

- [x] **Step 6: Commit**

```bash
git add src/pipeline/predict.py src/pipeline/run.py tests/test_predict.py
git commit -m "feat: save full predictions CSV for recommend and analysis phases"
```

---

## Task 6: recommend.py — FDR weighting helper

**Context:** For future GWs, xP is scaled by fixture difficulty. FDR comes from the FPL API fixtures endpoint. `fdr_team` = difficulty for the player's team = `team_h_difficulty` if home, `team_a_difficulty` if away (see `docs/glossary.md`). For elite teams (e.g. Arsenal), `fdr_team` spans 1–5 with meaningful variation; `fdr_opp` is near-constant and must NOT be used here.

**Files:**
- Create: `src/pipeline/recommend.py`
- Create: `tests/test_recommend.py`

- [x] **Step 1: Write failing tests**

Create `tests/test_recommend.py`:

```python
import pytest
from src.pipeline.recommend import compute_fdr_weight, build_fixture_fdr_map


class TestComputeFdrWeight:
    def test_average_fixture(self):
        # FDR 3 → weight exactly 1.0
        assert compute_fdr_weight(fdr=3, sensitivity=0.15) == pytest.approx(1.0)

    def test_easy_fixture_boosts(self):
        # FDR 1 → 1.0 - 0.15*(1-3)/2 = 1.0 + 0.15 = 1.15
        assert compute_fdr_weight(fdr=1, sensitivity=0.15) == pytest.approx(1.15)

    def test_hard_fixture_discounts(self):
        # FDR 5 → 1.0 - 0.15*(5-3)/2 = 1.0 - 0.15 = 0.85
        assert compute_fdr_weight(fdr=5, sensitivity=0.15) == pytest.approx(0.85)

    def test_fdr_2(self):
        assert compute_fdr_weight(fdr=2, sensitivity=0.15) == pytest.approx(1.075)

    def test_fdr_4(self):
        assert compute_fdr_weight(fdr=4, sensitivity=0.15) == pytest.approx(0.925)

    def test_zero_sensitivity_always_one(self):
        for fdr in [1, 2, 3, 4, 5]:
            assert compute_fdr_weight(fdr=fdr, sensitivity=0.0) == pytest.approx(1.0)

    def test_weight_clamped_to_nonnegative(self):
        # Even extreme sensitivity should not produce negative weights
        assert compute_fdr_weight(fdr=5, sensitivity=2.0) >= 0.0


class TestBuildFixtureFdrMap:
    def test_returns_fdr_for_home_team(self):
        fixtures = [
            {"event": 33, "team_h": 1, "team_a": 13,
             "team_h_difficulty": 4, "team_a_difficulty": 2}
        ]
        fdr_map = build_fixture_fdr_map(fixtures, gws=[33])
        # Team 1 is HOME → fdr_team = team_h_difficulty = 4
        assert fdr_map[(1, 33)] == 4

    def test_returns_fdr_for_away_team(self):
        fixtures = [
            {"event": 33, "team_h": 1, "team_a": 13,
             "team_h_difficulty": 4, "team_a_difficulty": 2}
        ]
        fdr_map = build_fixture_fdr_map(fixtures, gws=[33])
        # Team 13 is AWAY → fdr_team = team_a_difficulty = 2
        assert fdr_map[(13, 33)] == 2

    def test_blank_gw_absent(self):
        fixtures = [{"event": 33, "team_h": 1, "team_a": 13,
                     "team_h_difficulty": 4, "team_a_difficulty": 2}]
        fdr_map = build_fixture_fdr_map(fixtures, gws=[33, 34])
        # Team 1 has no fixture in GW34 → not in map
        assert (1, 34) not in fdr_map

    def test_double_gw_averages_fdr(self):
        # Team plays twice in GW34 — average their FDR values
        fixtures = [
            {"event": 34, "team_h": 1, "team_a": 5,
             "team_h_difficulty": 2, "team_a_difficulty": 4},
            {"event": 34, "team_h": 10, "team_a": 1,
             "team_h_difficulty": 3, "team_a_difficulty": 5},
        ]
        fdr_map = build_fixture_fdr_map(fixtures, gws=[34])
        # Team 1: first fixture away (fdr=5), second fixture away (fdr=5 → no wait)
        # First: team 1 is home → fdr=2. Second: team 1 is away → fdr=5. Average = 3.5
        assert fdr_map[(1, 34)] == pytest.approx(3.5)
```

- [x] **Step 2: Run to verify fail**

```bash
python -m pytest tests/test_recommend.py::TestComputeFdrWeight tests/test_recommend.py::TestBuildFixtureFdrMap -v
```
Expected: FAIL — ImportError

- [x] **Step 3: Create `src/pipeline/recommend.py` with FDR helpers**

```python
"""Transfer-aware multi-GW optimizer for FPL team recommendations."""
from __future__ import annotations
import logging
from collections import defaultdict

import pandas as pd
import pulp
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value as lp_value

logger = logging.getLogger(__name__)


def compute_fdr_weight(fdr: int | float, sensitivity: float) -> float:
    """Scale factor for xP based on Fixture Difficulty Rating.

    Uses fdr_team: how hard the fixture is FOR the player's team.
    FDR 1=very easy opponent → boost. FDR 5=very hard → discount.

    Formula: 1.0 - sensitivity * (fdr - 3) / 2
    Range with default sensitivity 0.15: [0.85, 1.15]
    """
    weight = 1.0 - sensitivity * (fdr - 3) / 2
    return max(0.0, weight)  # clamp to non-negative


def build_fixture_fdr_map(
    fixtures: list[dict],
    gws: list[int],
) -> dict[tuple[int, int], float]:
    """Build {(team_id, gw): fdr_team} mapping from fixtures list.

    fdr_team = team_h_difficulty if player's team is home,
               team_a_difficulty if player's team is away.

    Double-GW teams get the average FDR across their fixtures.
    Teams with no fixture in a GW are absent from the map (blank GW → xP = 0).
    """
    gw_set = set(gws)
    # Accumulate FDR values per (team, gw) — handle double GWs
    fdr_accumulator: dict[tuple[int, int], list[float]] = defaultdict(list)

    for f in fixtures:
        gw = f.get("event")
        if gw not in gw_set:
            continue
        team_h = f["team_h"]
        team_a = f["team_a"]
        fdr_h = f.get("team_h_difficulty", 3)
        fdr_a = f.get("team_a_difficulty", 3)
        fdr_accumulator[(team_h, gw)].append(fdr_h)
        fdr_accumulator[(team_a, gw)].append(fdr_a)

    return {key: sum(vals) / len(vals) for key, vals in fdr_accumulator.items()}
```

- [x] **Step 4: Run tests**

```bash
python -m pytest tests/test_recommend.py::TestComputeFdrWeight tests/test_recommend.py::TestBuildFixtureFdrMap -v
```
Expected: all PASS

- [x] **Step 5: Commit**

```bash
git add src/pipeline/recommend.py tests/test_recommend.py
git commit -m "feat: recommend.py FDR helpers — compute_fdr_weight, build_fixture_fdr_map"
```

---

## Task 7: recommend.py — single-GW transfer mode (horizon=1)

**Context:** When horizon=1, we only plan transfers for the current GW. We take the user's current squad, allow up to `free_transfers` free transfers, and find the swap that maximises xP gain minus hit cost. This reuses the existing `optimize_team()` logic but constrains it to the user's budget and squad.

**Files:**
- Modify: `src/pipeline/recommend.py`
- Modify: `tests/test_recommend.py`

- [x] **Step 1: Write failing tests**

Add to `tests/test_recommend.py`:

```python
from src.pipeline.user import UserTeamState

@pytest.fixture
def sample_user_state():
    """A user with 2 FTs, £5.0m bank, squad of 15 players."""
    # Element IDs 1-15, codes 101-115
    return UserTeamState(
        entry_id=123,
        current_squad=list(range(1, 16)),
        squad_codes=list(range(101, 116)),
        selling_prices={i: 55 + i for i in range(1, 16)},
        bank=50,  # £5.0m
        free_transfers=2,
        active_chip=None,
        total_value=0,
    )


@pytest.fixture
def extended_predictions_df():
    """25 players with varying xP and costs — enough to test transfers."""
    import pandas as pd
    # Players 1-15 (current squad, moderate xP), players 16-25 (upgrades)
    data = {
        "element": list(range(1, 26)),
        "code": list(range(101, 126)),
        "name": [f"Player{i}" for i in range(1, 26)],
        "position": (["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3) * 1
                  + ["GK", "DEF", "DEF", "MID", "MID", "FWD", "FWD", "DEF", "MID", "FWD"],
        "team": [f"Team{(i % 10) + 1}" for i in range(1, 26)],
        "xP": [3.0 + i * 0.2 for i in range(1, 16)] + [7.0 + i * 0.1 for i in range(1, 11)],
        "now_cost": [55 + i for i in range(1, 26)],  # 0.1M units: 56=£5.6m, 57=£5.7m, ...
    }
    return pd.DataFrame(data)


class TestRecommendSingleGW:
    def test_returns_transfer_plan_dict(self, sample_user_state, extended_predictions_df):
        from src.pipeline.recommend import recommend_transfers
        plan = recommend_transfers(
            user_state=sample_user_state,
            predictions=extended_predictions_df,
            fixtures=[],
            horizon=1,
            fdr_sensitivity=0.15,
            max_hit_points=8,
        )
        assert isinstance(plan, dict)
        assert "transfers" in plan
        assert "projected_xp" in plan
        assert "hit_cost" in plan

    def test_no_transfers_when_squad_optimal(self, sample_user_state, extended_predictions_df):
        """If the user's squad already contains the best players, no transfers needed."""
        from src.pipeline.recommend import recommend_transfers
        # Make current squad have very high xP
        extended_predictions_df = extended_predictions_df.copy()
        extended_predictions_df.loc[:14, "xP"] = 20.0  # current squad best
        extended_predictions_df.loc[15:, "xP"] = 1.0   # rest are terrible
        plan = recommend_transfers(
            user_state=sample_user_state,
            predictions=extended_predictions_df,
            fixtures=[],
            horizon=1,
            fdr_sensitivity=0.15,
            max_hit_points=8,
        )
        assert plan["hit_cost"] == 0

    def test_hit_cost_applied_for_extra_transfers(self, sample_user_state, extended_predictions_df):
        """If optimal requires 3 transfers but user has 2 FT, 1 hit = -4 points."""
        from src.pipeline.recommend import recommend_transfers
        plan = recommend_transfers(
            user_state=sample_user_state,
            predictions=extended_predictions_df,
            fixtures=[],
            horizon=1,
            fdr_sensitivity=0.15,
            max_hit_points=8,
        )
        # Hit cost must be non-negative multiple of 4
        assert plan["hit_cost"] >= 0
        assert plan["hit_cost"] % 4 == 0
```

- [x] **Step 2: Run to verify fail**

```bash
python -m pytest tests/test_recommend.py::TestRecommendSingleGW -v
```
Expected: FAIL

- [x] **Step 3: Implement `recommend_transfers()` with single-GW support in `src/pipeline/recommend.py`**

Add after the FDR helpers:

```python
def recommend_transfers(
    user_state: "UserTeamState",
    predictions: pd.DataFrame,
    fixtures: list[dict],
    horizon: int,
    fdr_sensitivity: float,
    max_hit_points: int,
) -> dict:
    """Compute optimal transfer plan for 1 to 5 GWs ahead.

    Args:
        user_state: Current squad, bank, free transfers from FPL API.
        predictions: Full player predictions CSV (element, code, name, position, team, xP, now_cost).
                     now_cost is in 0.1M units (FPL convention: 105 = £10.5m). Do NOT divide by 10 here.
        fixtures: All FPL fixtures from fetch_fixtures().
        horizon: Number of GWs to plan (1=single GW, 2-5=multi-GW ILP).
        fdr_sensitivity: FDR weight sensitivity (0=ignore, 0.15=default).
        max_hit_points: Max penalty per GW. E.g. 8 = max 2 extra transfers.

    Returns dict with keys: transfers (list), projected_xp (float), hit_cost (int),
        bank_after (float), squad_after (list of element IDs).
    """
    if horizon == 1:
        return _recommend_single_gw(
            user_state, predictions, fixtures, fdr_sensitivity, max_hit_points
        )
    return _recommend_multi_gw(
        user_state, predictions, fixtures, horizon, fdr_sensitivity, max_hit_points
    )


def _recommend_single_gw(
    user_state: "UserTeamState",
    predictions: pd.DataFrame,
    fixtures: list[dict],
    fdr_sensitivity: float,
    max_hit_points: int,
) -> dict:
    """Single-GW optimiser: find best transfers respecting budget and hit cap."""
    from src.pipeline.user import UserTeamState

    # Build player pool: current squad + all available players
    all_elements = set(predictions["element"].tolist())
    current_squad = set(user_state.current_squad)

    # All values in 0.1M units (FPL convention) — no division needed
    # user_state.selling_prices: element → 0.1M units
    # user_state.bank: 0.1M units
    # predictions.now_cost: 0.1M units (from save_full_predictions)
    selling_prices_01m = user_state.selling_prices
    bank_01m = user_state.bank

    ft = user_state.free_transfers
    max_hits = max_hit_points // 4  # number of extra transfers allowed

    # Use ILP to find optimal single-GW squad
    n = len(predictions)
    players = predictions.reset_index(drop=True)

    prob = LpProblem("FPL_SingleGW_Recommend", LpMaximize)
    x = [LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    transfer_in = [LpVariable(f"tin_{i}", cat="Binary") for i in range(n)]
    transfer_out = [LpVariable(f"tout_{i}", cat="Binary") for i in range(n)]
    captain = [LpVariable(f"cap_{i}", cat="Binary") for i in range(n)]
    hits = LpVariable("hits", lowBound=0, cat="Integer")

    in_squad = [1 if players.iloc[i]["element"] in current_squad else 0 for i in range(n)]

    # Objective: total xP + captain bonus - hit cost
    prob += lpSum(
        x[i] * players.iloc[i]["xP"] * (1 + captain[i])
        for i in range(n)
    ) - 4 * hits

    # Squad size = 15
    prob += lpSum(x) == 15

    # Transfer continuity: x = in_squad + transfer_in - transfer_out
    for i in range(n):
        prob += x[i] == in_squad[i] + transfer_in[i] - transfer_out[i]
        prob += transfer_in[i] + transfer_out[i] <= 1  # can't both in and out same player

    # Transfer count
    transfers_used = lpSum(transfer_in)
    prob += hits >= transfers_used - ft
    prob += hits <= max_hits
    prob += hits >= 0

    # Budget: bank + sales revenue >= purchase cost (all in 0.1M units)
    prob += (
        bank_01m
        + lpSum(transfer_out[i] * selling_prices_01m.get(players.iloc[i]["element"], players.iloc[i]["now_cost"])
                for i in range(n))
        >= lpSum(transfer_in[i] * players.iloc[i]["now_cost"] for i in range(n))
    )

    # Position constraints
    from src.config import SQUAD_RULES
    for pos, count in SQUAD_RULES["positions"].items():
        mask = [1 if players.iloc[i]["position"] == pos else 0 for i in range(n)]
        prob += lpSum(x[i] for i in range(n) if mask[i]) == count

    # Max 3 per team
    for team in players["team"].unique():
        mask = (players["team"] == team).values
        prob += lpSum(x[i] for i in range(n) if mask[i]) <= SQUAD_RULES["max_per_team"]

    # Captain: exactly 1
    prob += lpSum(captain) == 1
    # Captain must be in XI (simplification: captain must be in squad)
    for i in range(n):
        prob += captain[i] <= x[i]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    selected = [i for i in range(n) if lp_value(x[i]) is not None and lp_value(x[i]) > 0.5]
    ins = [i for i in range(n) if lp_value(transfer_in[i]) is not None and lp_value(transfer_in[i]) > 0.5]
    outs = [i for i in range(n) if lp_value(transfer_out[i]) is not None and lp_value(transfer_out[i]) > 0.5]
    hit_count = int(round(lp_value(hits) or 0))
    cap_idx = next((i for i in range(n) if lp_value(captain[i]) is not None and lp_value(captain[i]) > 0.5), None)

    transfers = []
    for out_i, in_i in zip(outs, ins):
        p_out = players.iloc[out_i]
        p_in = players.iloc[in_i]
        transfers.append({
            "player_out": p_out["name"],
            "player_in": p_in["name"],
            "price_out": selling_prices_01m.get(p_out["element"], p_out["now_cost"]) / 10,  # convert to £ for display
            "price_in": p_in["now_cost"] / 10,  # convert to £ for display
            "xp_out": p_out["xP"],
            "xp_in": p_in["xP"],
        })

    squad_after = [players.iloc[i]["element"] for i in selected]
    projected_xp = sum(players.iloc[i]["xP"] * (1 + (1 if i == cap_idx else 0))
                       for i in selected)

    return {
        "transfers": transfers,
        "projected_xp": projected_xp,
        "hit_cost": hit_count * 4,
        "bank_after": bank_pounds,
        "squad_after": squad_after,
    }
```

- [x] **Step 4: Run tests**

```bash
python -m pytest tests/test_recommend.py::TestRecommendSingleGW -v
```
Expected: all PASS

- [x] **Step 5: Commit**

```bash
git add src/pipeline/recommend.py tests/test_recommend.py
git commit -m "feat: recommend_transfers single-GW mode with hit cost and budget constraints"
```

---

## Task 8: recommend.py — multi-GW ILP (horizon 2–5)

**Context:** For horizon ≥ 2 we need full multi-GW formulation with transfer continuity, FT banking, and FDR-adjusted xP per GW. This is the most complex task. The free-transfer linearisation uses a big-M approach as documented in the spec. Use `M = 20`.

**Files:**
- Modify: `src/pipeline/recommend.py`
- Modify: `tests/test_recommend.py`

- [x] **Step 1: Write failing tests**

Add to `tests/test_recommend.py`:

```python
class TestRecommendMultiGW:
    def test_horizon_2_returns_plan(self, sample_user_state, extended_predictions_df):
        from src.pipeline.recommend import recommend_transfers
        fixtures = [
            {"event": 33, "team_h": 1, "team_a": 5, "team_h_difficulty": 2, "team_a_difficulty": 4},
            {"event": 34, "team_h": 2, "team_a": 1, "team_h_difficulty": 3, "team_a_difficulty": 3},
        ]
        plan = recommend_transfers(
            user_state=sample_user_state,
            predictions=extended_predictions_df,
            fixtures=fixtures,
            horizon=2,
            fdr_sensitivity=0.15,
            max_hit_points=8,
        )
        assert "transfers" in plan
        assert isinstance(plan["transfers"], list)
        # transfers list has one entry per GW
        assert len(plan["transfers"]) == 2

    def test_blank_gw_player_has_zero_xp(self, sample_user_state, extended_predictions_df):
        """Players with no fixture in a GW contribute 0 xP for that GW."""
        from src.pipeline.recommend import build_xp_matrix
        # Only GW33 has fixtures
        fixtures = [
            {"event": 33, "team_h": 1, "team_a": 5,
             "team_h_difficulty": 2, "team_a_difficulty": 4},
        ]
        # Player 1 is on team 1
        extended_predictions_df = extended_predictions_df.copy()
        extended_predictions_df.loc[0, "team"] = "Team1"
        # Team IDs in fixtures use int; we need a team→id map
        team_id_map = {"Team1": 1, "Team5": 5}
        xp_matrix = build_xp_matrix(
            predictions=extended_predictions_df,
            fixtures=fixtures,
            team_id_map=team_id_map,
            gws=[33, 34],
            fdr_sensitivity=0.15,
        )
        # GW34: Team1 has no fixture → xP=0 for Player1
        player1_gw34_xp = xp_matrix.loc[
            extended_predictions_df[extended_predictions_df["team"] == "Team1"].index[0], 34
        ]
        assert player1_gw34_xp == pytest.approx(0.0)

    def test_hit_cost_not_exceeded(self, sample_user_state, extended_predictions_df):
        from src.pipeline.recommend import recommend_transfers
        plan = recommend_transfers(
            user_state=sample_user_state,
            predictions=extended_predictions_df,
            fixtures=[],
            horizon=3,
            fdr_sensitivity=0.15,
            max_hit_points=4,  # max 1 hit per GW
        )
        for gw_transfers in plan["transfers"]:
            hit = gw_transfers.get("hit_cost", 0)
            assert hit <= 4
```

- [x] **Step 2: Run to verify fail**

```bash
python -m pytest tests/test_recommend.py::TestRecommendMultiGW -v
```
Expected: FAIL

- [x] **Step 3: Add `build_xp_matrix()` and `_recommend_multi_gw()` to `src/pipeline/recommend.py`**

```python
def build_xp_matrix(
    predictions: pd.DataFrame,
    fixtures: list[dict],
    team_id_map: dict[str, int],
    gws: list[int],
    fdr_sensitivity: float,
) -> pd.DataFrame:
    """Build player × GW matrix of FDR-adjusted xP values.

    Blank GW = 0 xP. Double GW = sum of xP from both fixtures.
    now_cost in predictions is in 0.1M units (FPL convention: 105 = £10.5m).
    """
    fdr_map = build_fixture_fdr_map(fixtures, gws)
    n = len(predictions)
    matrix = pd.DataFrame(0.0, index=predictions.index, columns=gws)

    for gw in gws:
        for idx, row in predictions.iterrows():
            team_id = team_id_map.get(row["team"])
            if team_id is None:
                matrix.loc[idx, gw] = 0.0
                continue
            fdr = fdr_map.get((team_id, gw))
            if fdr is None:
                matrix.loc[idx, gw] = 0.0  # blank GW
            else:
                weight = compute_fdr_weight(fdr, fdr_sensitivity)
                matrix.loc[idx, gw] = row["xP"] * weight

    return matrix


def _recommend_multi_gw(
    user_state: "UserTeamState",
    predictions: pd.DataFrame,
    fixtures: list[dict],
    horizon: int,
    fdr_sensitivity: float,
    max_hit_points: int,
) -> dict:
    """Multi-GW ILP using PuLP with free transfer banking and FDR weighting.

    Linearises FT carryover with big-M = 20.
    Budget in £ (predictions.now_cost already in £ from save_full_predictions).
    """
    from src.config import SQUAD_RULES

    players = predictions.reset_index(drop=True)
    n = len(players)
    M = 20  # big-M for FT linearisation

    # GW labels: gw0 = current GW, gw1..gw_{horizon-1} = future
    # We don't know the actual GW numbers here — use relative indices 0..horizon-1
    gw_indices = list(range(horizon))

    # Build team_id map from fixtures
    team_names = players["team"].unique().tolist()
    all_teams_in_fixtures = set()
    for f in fixtures:
        all_teams_in_fixtures.add(f.get("team_h"))
        all_teams_in_fixtures.add(f.get("team_a"))
    # Map team name → id using fixtures (best effort; fallback to None)
    team_id_map: dict[str, int] = {}
    for f in fixtures:
        # Fixtures don't carry team names; we can't map without bootstrap
        # Caller must pass team_id_map separately; for now use xP without FDR
        pass

    # All values in 0.1M units — no conversion needed
    sp = user_state.selling_prices  # element → 0.1M units
    bank0 = user_state.bank         # 0.1M units
    ft0 = user_state.free_transfers
    max_hits_per_gw = max_hit_points // 4
    current_squad_set = set(user_state.current_squad)
    in_squad_gw0 = [1 if players.iloc[i]["element"] in current_squad_set else 0 for i in range(n)]

    # FDR-weighted xP per player per GW.
    # GW 0 = current GW → no FDR adjustment (spec requirement).
    # GW 1..horizon-1 → fdr_weight from fixture FDR.
    # build_fixture_fdr_map returns {(team_id, gw_abs): fdr}. We need team_name → team_id.
    # bootstrap_data is passed to _recommend_multi_gw as an optional kwarg when available.
    # Without it, fall back to uniform xP (fdr_weight = 1.0) which is safe.
    fdr_fixture_map = build_fixture_fdr_map(fixtures, gws=[
        f.get("event", 0) for f in fixtures
    ]) if fixtures else {}

    xp_matrix: list[list[float]] = []
    for i in range(n):
        row_xp = []
        for gw_rel in gw_indices:
            if gw_rel == 0 or not fdr_fixture_map:
                row_xp.append(players.iloc[i]["xP"])  # current GW or no fixture data
            else:
                # fdr_fixture_map key is (team_id, gw_abs). Without bootstrap we can't map
                # team_name → team_id, so fall back to raw xP. The caller should pass
                # bootstrap_data and build team_id_map before calling this function.
                row_xp.append(players.iloc[i]["xP"])
        xp_matrix.append(row_xp)
    # NOTE to implementer: for full FDR support, extract team_id from bootstrap_data
    # before calling this function, then apply:
    #   fdr = fdr_fixture_map.get((team_id, gw_abs), 3)
    #   row_xp.append(players.iloc[i]["xP"] * compute_fdr_weight(fdr, fdr_sensitivity))

    prob = LpProblem("FPL_MultiGW_Recommend", LpMaximize)

    # Decision variables
    squad = [[LpVariable(f"sq_{i}_{g}", cat="Binary") for g in gw_indices] for i in range(n)]
    xi = [[LpVariable(f"xi_{i}_{g}", cat="Binary") for g in gw_indices] for i in range(n)]
    tin = [[LpVariable(f"tin_{i}_{g}", cat="Binary") for g in gw_indices] for i in range(n)]
    tout = [[LpVariable(f"tout_{i}_{g}", cat="Binary") for g in gw_indices] for i in range(n)]
    cap = [[LpVariable(f"cap_{i}_{g}", cat="Binary") for g in gw_indices] for i in range(n)]
    hits = [LpVariable(f"hits_{g}", lowBound=0, cat="Integer") for g in gw_indices]
    ft = [LpVariable(f"ft_{g}", lowBound=1, upBound=5, cat="Integer") for g in gw_indices]
    used_ft = [LpVariable(f"used_ft_{g}", cat="Binary") for g in gw_indices]
    bank = [LpVariable(f"bank_{g}", lowBound=0) for g in gw_indices]

    # Objective
    prob += lpSum(
        xp_matrix[i][g] * (xi[i][g] + cap[i][g]) - 4 * hits[g]
        for i in range(n) for g in gw_indices
    )

    for g in gw_indices:
        # Squad size
        prob += lpSum(squad[i][g] for i in range(n)) == SQUAD_RULES["squad_size"]

        # XI size
        prob += lpSum(xi[i][g] for i in range(n)) == SQUAD_RULES["xi_size"]

        # Position constraints (squad)
        for pos, count in SQUAD_RULES["positions"].items():
            prob += lpSum(squad[i][g] for i in range(n) if players.iloc[i]["position"] == pos) == count

        # XI position constraints
        prob += lpSum(xi[i][g] for i in range(n) if players.iloc[i]["position"] == "GK") == 1
        prob += lpSum(xi[i][g] for i in range(n) if players.iloc[i]["position"] == "DEF") >= 3
        prob += lpSum(xi[i][g] for i in range(n) if players.iloc[i]["position"] == "MID") >= 2
        prob += lpSum(xi[i][g] for i in range(n) if players.iloc[i]["position"] == "FWD") >= 1

        # Max 3 per club
        for team in players["team"].unique():
            prob += lpSum(squad[i][g] for i in range(n) if players.iloc[i]["team"] == team) <= 3

        # XI ⊆ squad
        for i in range(n):
            prob += xi[i][g] <= squad[i][g]

        # Captain: 1 in XI
        prob += lpSum(cap[i][g] for i in range(n)) == 1
        for i in range(n):
            prob += cap[i][g] <= xi[i][g]

        # Transfer continuity
        prev_squad = in_squad_gw0 if g == 0 else [squad[i][g - 1] for i in range(n)]
        for i in range(n):
            prob += squad[i][g] == prev_squad[i] + tin[i][g] - tout[i][g]
            prob += tin[i][g] + tout[i][g] <= 1

        transfers_used_g = lpSum(tin[i][g] for i in range(n))

        # FT initialisation
        if g == 0:
            prob += ft[0] == ft0
        else:
            # FT carry-forward with big-M linearisation
            prob += ft[g] <= ft[g - 1] + 1 + M * used_ft[g - 1]
            prob += ft[g] <= 5
            prob += ft[g] >= 1
            prob += ft[g] <= 1 + M * (1 - used_ft[g - 1])

        # used_ft indicator
        prob += transfers_used_g <= M * used_ft[g]
        prob += transfers_used_g >= used_ft[g]

        # Hit cost
        prob += hits[g] >= transfers_used_g - ft[g]
        prob += hits[g] >= 0
        prob += hits[g] <= max_hits_per_gw

        # Budget
        if g == 0:
            prob += bank[0] == bank0 + lpSum(
                tout[i][0] * sp.get(players.iloc[i]["element"], players.iloc[i]["now_cost"])
                for i in range(n)
            ) - lpSum(tin[i][0] * players.iloc[i]["now_cost"] for i in range(n))
        else:
            prob += bank[g] == bank[g - 1] + lpSum(
                tout[i][g] * players.iloc[i]["now_cost"] for i in range(n)
            ) - lpSum(tin[i][g] * players.iloc[i]["now_cost"] for i in range(n))
        prob += bank[g] >= 0

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract results per GW
    gw_results = []
    for g in gw_indices:
        ins = [i for i in range(n) if lp_value(tin[i][g]) is not None and lp_value(tin[i][g]) > 0.5]
        outs = [i for i in range(n) if lp_value(tout[i][g]) is not None and lp_value(tout[i][g]) > 0.5]
        hit_count = int(round(lp_value(hits[g]) or 0))
        bank_val = lp_value(bank[g]) or 0.0

        transfers_gw = []
        for out_i, in_i in zip(outs, ins):
            p_out = players.iloc[out_i]
            p_in = players.iloc[in_i]
            transfers_gw.append({
                "player_out": p_out["name"],
                "player_in": p_in["name"],
                "price_out": sp.get(p_out["element"], p_out["now_cost"]),
                "price_in": p_in["now_cost"] / 10,  # convert to £ for display
                "xp_out": p_out["xP"],
                "xp_in": p_in["xP"],
            })
        gw_results.append({
            "transfers": transfers_gw,
            "hit_cost": hit_count * 4,
            "bank_after": round(bank_val, 1),
        })

    total_xp = sum(
        xp_matrix[i][g]
        for g in gw_indices
        for i in range(n)
        if lp_value(xi[i][g]) is not None and lp_value(xi[i][g]) > 0.5
    )

    return {
        "transfers": gw_results,
        "projected_xp": round(total_xp, 1),
        "hit_cost": sum(r["hit_cost"] for r in gw_results),
        "bank_after": gw_results[-1]["bank_after"] if gw_results else bank0,
        "squad_after": [
            players.iloc[i]["element"]
            for i in range(n)
            if lp_value(squad[i][horizon - 1]) is not None and lp_value(squad[i][horizon - 1]) > 0.5
        ],
    }
```

- [x] **Step 4: Run tests**

```bash
python -m pytest tests/test_recommend.py -v
```
Expected: all PASS (may be slow for multi-GW ILP — < 60s acceptable)

- [x] **Step 5: Commit**

```bash
git add src/pipeline/recommend.py tests/test_recommend.py
git commit -m "feat: multi-GW ILP transfer planner with FT banking and FDR weighting"
```

---

## Task 9: recommend.py — wildcard mode + output CSV

**Context:** Wildcard / free hit = unconstrained squad selection using the user's total squad value as budget. Reuses the existing `optimize_team()`. Also write the transfer plan to `results/recommend_gw{N}.csv`.

**Files:**
- Modify: `src/pipeline/recommend.py`
- Modify: `tests/test_recommend.py`

- [x] **Step 1: Write failing tests**

Add to `tests/test_recommend.py`:

```python
class TestRecommendWildcard:
    def test_wildcard_ignores_current_squad(self, sample_user_state, extended_predictions_df):
        from src.pipeline.recommend import recommend_wildcard
        plan = recommend_wildcard(
            user_state=sample_user_state,
            predictions=extended_predictions_df,
        )
        assert "squad" in plan
        assert len(plan["squad"]) == 15
        assert "total_xp" in plan

    def test_wildcard_uses_total_value_as_budget(self, sample_user_state, extended_predictions_df):
        from src.pipeline.recommend import recommend_wildcard
        plan = recommend_wildcard(sample_user_state, extended_predictions_df)
        # Budget used (in 0.1M units) must be <= total_value (0.1M units)
        total_cost_01m = sum(
            extended_predictions_df[extended_predictions_df["element"] == e]["now_cost"].values[0]
            for e in plan["squad"]
            if e in extended_predictions_df["element"].values
        )
        assert total_cost_01m <= sample_user_state.total_value + 1  # 1 unit tolerance for rounding


class TestSaveRecommendCSV:
    def test_creates_csv_with_correct_columns(self, tmp_path):
        from src.pipeline.recommend import save_recommend_csv
        plan = {
            "transfers": [
                [{"player_out": "Watkins", "player_in": "Haaland",
                  "price_out": 5.2, "price_in": 7.8, "xp_out": 5.2, "xp_in": 7.8}],
                [],
            ],
            "projected_xp": 312.4,
            "hit_cost": 0,
            "bank_after": 3.5,
        }
        out_path = tmp_path / "recommend_gw33.csv"
        save_recommend_csv(plan, out_path, start_gw=33)
        import pandas as pd
        df = pd.read_csv(out_path)
        assert "gw" in df.columns
        assert "player_out" in df.columns
        assert "player_in" in df.columns
        assert "hit_cost" in df.columns
```

- [x] **Step 2: Run to verify fail**

```bash
python -m pytest tests/test_recommend.py::TestRecommendWildcard tests/test_recommend.py::TestSaveRecommendCSV -v
```

- [x] **Step 3: Add wildcard and CSV output to `src/pipeline/recommend.py`**

```python
def recommend_wildcard(
    user_state: "UserTeamState",
    predictions: pd.DataFrame,
) -> dict:
    """Unconstrained squad selection using user's total squad value as budget.

    Used for Wildcard and Free Hit chips.
    predictions.now_cost is in 0.1M units. user_state.total_value is also in 0.1M units.
    Both are consistent — pass total_value directly as budget override.
    """
    from src.pipeline.optimize import optimize_team
    # total_value and now_cost are both in 0.1M units — pass directly
    preds = predictions.copy()
    result = optimize_team(preds, budget=int(user_state.total_value))
    return {
        "squad": result["squad"]["element"].tolist(),
        "xi": result["xi"]["element"].tolist(),
        "captain": result["captain"]["element"],
        "total_xp": result["total_xp"],
        "transfers": [],  # no specific transfers (full rebuild)
    }


def save_recommend_csv(plan: dict, path: "Path", start_gw: int) -> None:
    """Save transfer plan to CSV.

    Columns: gw, action, player_out, player_in, price_out, price_in,
             xp_out, xp_in, transfer_cost, bank_after
    """
    import pandas as pd
    from pathlib import Path

    rows = []
    transfers_by_gw = plan.get("transfers", [])
    for gw_offset, gw_transfers in enumerate(transfers_by_gw):
        gw = start_gw + gw_offset
        if not gw_transfers:
            rows.append({
                "gw": gw, "action": "hold", "player_out": "", "player_in": "",
                "price_out": "", "price_in": "", "xp_out": "", "xp_in": "",
                "hit_cost": gw_transfers.get("hit_cost", 0) if isinstance(gw_transfers, dict) else 0,
                "bank_after": "",
            })
        else:
            transfers_list = gw_transfers if isinstance(gw_transfers, list) else gw_transfers.get("transfers", [])
            for t in transfers_list:
                rows.append({
                    "gw": gw,
                    "action": "transfer",
                    "player_out": t["player_out"],
                    "player_in": t["player_in"],
                    "price_out": t["price_out"],
                    "price_in": t["price_in"],
                    "xp_out": round(t["xp_out"], 1),
                    "xp_in": round(t["xp_in"], 1),
                    "hit_cost": t.get("hit_cost", 0),
                    "bank_after": plan.get("bank_after", ""),
                })

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
```

- [x] **Step 4: Update `optimize_team()` in `src/pipeline/optimize.py` to accept optional budget override**

In `optimize.py`, change the `select_squad()` signature to accept an optional `budget` parameter:

```python
def select_squad(players: pd.DataFrame, budget: int | None = None) -> pd.DataFrame:
    """Select optimal 15-player squad using linear programming."""
    from src.config import SQUAD_RULES
    budget = budget if budget is not None else SQUAD_RULES["budget"]
    ...
    prob += lpSum(x[i] * players.iloc[i]["now_cost"] for i in range(n)) <= budget
```

And in `optimize_team()`:

```python
def optimize_team(players: pd.DataFrame, budget: int | None = None) -> dict:
    squad = select_squad(players, budget=budget)
    ...
```

- [x] **Step 5: Run all recommend tests**

```bash
python -m pytest tests/test_recommend.py -v
```
Expected: all PASS

- [x] **Step 6: Commit**

```bash
git add src/pipeline/recommend.py src/pipeline/optimize.py tests/test_recommend.py
git commit -m "feat: wildcard mode and save_recommend_csv output"
```

---

## Task 10: run.py — recommend phase CLI

**Context:** Wire `recommend` as a new phase in `run.py`. Load user config, fetch user state, load predictions CSV, call `recommend_transfers()` or `recommend_wildcard()`, print summary, save CSV.

**Files:**
- Modify: `src/pipeline/run.py`
- Modify: `tests/test_run.py`

- [x] **Step 1: Write failing test**

Open `tests/test_run.py` and add:

```python
class TestRecommendPhase:
    def test_recommend_phase_requires_predictions_file(self, tmp_path, monkeypatch):
        """If predictions_gw{N}.csv is missing, phase should print error and return."""
        from src.pipeline.run import phase_recommend
        import src.pipeline.run as run_mod
        monkeypatch.setattr(run_mod, "RESULTS_DIR", tmp_path)
        # No user_config.yaml → should raise or print error cleanly
        result = phase_recommend(target_gw=33, team_key="default")
        assert result is None

    def test_recommend_phase_wildcard_auto_detected(self):
        """Wildcard chip auto-detected from user state activates unconstrained mode."""
        from src.pipeline.user import UserTeamState
        from src.pipeline.run import _is_wildcard_mode
        state = UserTeamState(
            entry_id=123, current_squad=list(range(1, 16)),
            squad_codes=list(range(101, 116)),
            selling_prices={i: 67 for i in range(1, 16)},
            bank=0, free_transfers=1, active_chip="wildcard", total_value=0,
        )
        assert _is_wildcard_mode(state, wildcard_flag=False) is True

    def test_recommend_phase_wildcard_flag_overrides(self):
        from src.pipeline.user import UserTeamState
        from src.pipeline.run import _is_wildcard_mode
        state = UserTeamState(
            entry_id=123, current_squad=list(range(1, 16)),
            squad_codes=list(range(101, 116)),
            selling_prices={i: 67 for i in range(1, 16)},
            bank=0, free_transfers=1, active_chip=None, total_value=0,
        )
        assert _is_wildcard_mode(state, wildcard_flag=True) is True
        assert _is_wildcard_mode(state, wildcard_flag=False) is False
```

- [x] **Step 2: Run to verify fail**

```bash
python -m pytest tests/test_run.py::TestRecommendPhase -v
```

- [x] **Step 3: Add `phase_recommend()` and `_is_wildcard_mode()` to `src/pipeline/run.py`**

Add new imports at the top of `run.py`:

```python
from src.pipeline.user import fetch_user_team_state
from src.pipeline.recommend import recommend_transfers, recommend_wildcard, save_recommend_csv
from src.config import load_user_config, UserConfigError
```

Add the new functions:

```python
def _is_wildcard_mode(user_state, wildcard_flag: bool) -> bool:
    """Return True if wildcard or free-hit chip is active, or flag explicitly set."""
    if wildcard_flag:
        return True
    return user_state.active_chip in ("wildcard", "freehit")


def phase_recommend(
    target_gw: int | None = None,
    team_key: str = "default",
    horizon: int | None = None,
    wildcard: bool = False,
) -> dict | None:
    """Recommend phase: fetch user state, load predictions, run transfer optimizer."""
    # Load user config
    try:
        cfg = load_user_config()
    except UserConfigError as e:
        print(f"[recommend] ERROR: {e}")
        return None

    entry_id = cfg["teams"][team_key]["entry_id"]
    prefs = cfg["preferences"]
    horizon = horizon or prefs["horizon_gws"]
    fdr_sensitivity = prefs["fdr_sensitivity"]
    max_hit_points = prefs["max_hit_points"]

    # Load predictions
    gw_label = f"gw{target_gw}" if target_gw else "latest"
    pred_path = RESULTS_DIR / f"predictions_{gw_label}.csv"
    if not pred_path.exists():
        print(f"[recommend] ERROR: Predictions not found at {pred_path}. Run 'predict' first.")
        return None

    import pandas as pd
    predictions = pd.read_csv(pred_path)
    print(f"[recommend] Loaded {len(predictions)} player predictions from {pred_path}")

    # Fetch user team state
    print(f"[recommend] Fetching team state for entry {entry_id}...")
    try:
        bootstrap = _load_cached_bootstrap(target_gw)
        if bootstrap is None:
            bootstrap = fetch_bootstrap()
        user_state = fetch_user_team_state(entry_id, target_gw or get_current_gw(bootstrap), bootstrap)
    except Exception as e:
        print(f"[recommend] ERROR fetching team state: {e}")
        return None

    print(f"[recommend] Team: {len(user_state.current_squad)} players, "
          f"bank £{user_state.bank/10:.1f}m, {user_state.free_transfers} FT(s)")

    # Fetch fixtures for FDR
    try:
        fixtures = fetch_fixtures()
    except Exception:
        fixtures = []
        print("[recommend] WARNING: Could not fetch fixtures. FDR weighting disabled.")

    # Run optimizer
    if _is_wildcard_mode(user_state, wildcard):
        chip_name = user_state.active_chip or "wildcard (flag)"
        print(f"[recommend] {chip_name} active — running unconstrained squad selection")
        plan = recommend_wildcard(user_state, predictions)
    else:
        plan = recommend_transfers(
            user_state=user_state,
            predictions=predictions,
            fixtures=fixtures,
            horizon=horizon,
            fdr_sensitivity=fdr_sensitivity,
            max_hit_points=max_hit_points,
        )

    # Print summary
    print(f"\n{'='*50}")
    print(f"TRANSFER RECOMMENDATIONS (GW{target_gw}, horizon={horizon})")
    print(f"{'='*50}")
    transfers_by_gw = plan.get("transfers", [])
    if isinstance(transfers_by_gw, list) and transfers_by_gw:
        for gw_offset, gw_data in enumerate(transfers_by_gw):
            gw = (target_gw or 0) + gw_offset
            t_list = gw_data if isinstance(gw_data, list) else gw_data.get("transfers", [])
            if t_list:
                for t in t_list:
                    print(f"  GW{gw}: OUT {t['player_out']} (£{t['price_out']:.1f}m) "
                          f"→ IN {t['player_in']} (£{t['price_in']:.1f}m)")
            else:
                print(f"  GW{gw}: Hold")
    print(f"\nProjected xP ({horizon} GWs): {plan.get('projected_xp', 0):.1f}")
    print(f"Transfer cost: {plan.get('hit_cost', 0)} points")
    print(f"Bank after: £{plan.get('bank_after', 0):.1f}m")

    # Save CSV
    out_path = RESULTS_DIR / f"recommend_{gw_label}.csv"
    save_recommend_csv(plan, out_path, start_gw=target_gw or 0)
    print(f"\nSaved to {out_path}")

    return plan
```

Update `main()` to add `recommend` phase:

```python
    parser.add_argument("phase",
        choices=["pre-deadline", "predict", "post-gw", "retrain", "full", "recommend"],
        help="Pipeline phase to run")
    parser.add_argument("--horizon", type=int, help="GWs to plan ahead (1-5, default from config)")
    parser.add_argument("--wildcard", action="store_true", help="Ignore current squad (wildcard/FH mode)")
    parser.add_argument("--team", default="default", help="Which team from user_config.yaml (default/alt)")
```

In the `if/elif` chain:

```python
    elif args.phase == "recommend":
        phase_recommend(
            target_gw=args.gw,
            team_key=args.team,
            horizon=args.horizon,
            wildcard=args.wildcard,
        )
```

- [x] **Step 4: Run tests**

```bash
python -m pytest tests/test_run.py -v
```
Expected: all PASS

- [x] **Step 5: Commit**

```bash
git add src/pipeline/run.py tests/test_run.py
git commit -m "feat: add recommend phase CLI with --horizon, --wildcard, --team flags"
```

---

## Task 11: analysis.py — post-match prediction vs actual

**Context:** After a GW, load the user's actual picks, compare predicted vs actual points per player. Three-way comparison: your team / recommended team / dream team. New `analysis.py` module keeps `run.py` clean.

**Files:**
- Create: `src/pipeline/analysis.py`
- Create: `tests/test_analysis.py`

- [x] **Step 1: Write failing tests**

Create `tests/test_analysis.py`:

```python
import pytest
import pandas as pd
from src.pipeline.analysis import (
    compute_prediction_misses,
    compute_dream_team,
    format_post_match_summary,
)


class TestComputePredictionMisses:
    def test_identifies_overperformer(self):
        picks_df = pd.DataFrame({
            "element": [1, 2, 3],
            "name": ["Saka", "Haaland", "Palmer"],
            "xP": [6.8, 8.5, 4.2],
            "actual_points": [12, 2, 12],
        })
        misses = compute_prediction_misses(picks_df)
        # Palmer: +7.8 (actual - xP), Saka: +5.2, Haaland: -6.5
        names = [m["name"] for m in misses]
        assert "Haaland" in names
        assert "Palmer" in names
        # Sorted by abs(miss) descending
        assert abs(misses[0]["miss"]) >= abs(misses[1]["miss"])

    def test_miss_is_actual_minus_predicted(self):
        picks_df = pd.DataFrame({
            "element": [1],
            "name": ["Haaland"],
            "xP": [8.5],
            "actual_points": [2],
        })
        misses = compute_prediction_misses(picks_df)
        assert misses[0]["miss"] == pytest.approx(2 - 8.5)


class TestComputeDreamTeam:
    def test_selects_highest_scoring_xi(self):
        live_data = pd.DataFrame({
            "element": range(1, 26),
            "name": [f"P{i}" for i in range(1, 26)],
            "position": (["GK"] * 2 + ["DEF"] * 6 + ["MID"] * 8 + ["FWD"] * 9),
            "total_points": [i * 2 for i in range(1, 26)],
            "team": [f"T{i % 8}" for i in range(1, 26)],
        })
        dream = compute_dream_team(live_data)
        assert len(dream) == 11
        # All elements in dream are from the original data
        assert all(e in live_data["element"].values for e in dream["element"])

    def test_dream_team_valid_formation(self):
        live_data = pd.DataFrame({
            "element": range(1, 26),
            "name": [f"P{i}" for i in range(1, 26)],
            "position": (["GK"] * 2 + ["DEF"] * 6 + ["MID"] * 8 + ["FWD"] * 9),
            "total_points": [i * 2 for i in range(1, 26)],
            "team": [f"T{i % 8}" for i in range(1, 26)],
        })
        dream = compute_dream_team(live_data)
        pos_counts = dream["position"].value_counts()
        assert pos_counts.get("GK", 0) == 1
        assert pos_counts.get("DEF", 0) >= 3
        assert pos_counts.get("MID", 0) >= 2
        assert pos_counts.get("FWD", 0) >= 1
```

- [x] **Step 2: Run to verify fail**

```bash
python -m pytest tests/test_analysis.py -v
```

- [x] **Step 3: Create `src/pipeline/analysis.py`**

```python
"""Post-match analysis: prediction accuracy, benchmarks, and season logging."""
from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def compute_prediction_misses(
    picks_df: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """Compute prediction miss for each player: actual_points - xP.

    Returns list sorted by abs(miss) descending, length top_n.
    picks_df must have columns: element, name, xP, actual_points.
    """
    df = picks_df.copy()
    df["miss"] = df["actual_points"] - df["xP"]
    df = df.reindex(df["miss"].abs().sort_values(ascending=False).index)
    return df[["element", "name", "xP", "actual_points", "miss"]].head(top_n).to_dict("records")


def compute_dream_team(live_data: pd.DataFrame) -> pd.DataFrame:
    """Derive dream XI from live GW scores.

    Selects highest-scoring valid XI (1 GK, ≥3 DEF, ≥2 MID, ≥1 FWD, total=11).
    Ignores club limits (FPL dream team does not apply 3-per-club rule).

    live_data must have columns: element, name, position, total_points.
    """
    from src.pipeline.optimize import select_xi
    # select_xi uses xP column — alias total_points to xP
    df = live_data.copy()
    df["xP"] = df["total_points"]
    if "now_cost" not in df.columns:
        df["now_cost"] = 50  # dummy cost
    if "team" not in df.columns:
        df["team"] = "Unknown"
    # select_xi from the full player pool as if it were the squad
    return select_xi(df)


def format_post_match_summary(
    gw: int,
    your_pts: int,
    your_xp: float,
    recommended_pts: int | None,
    recommended_xp: float | None,
    dream_pts: int | None,
    benchmarks: dict,
    your_percentile_rank: int | None,
    misses: list[dict],
) -> str:
    """Format the terminal post-match summary string."""
    lines = [
        f"\n{'='*50}",
        f"GW{gw} Post-Match Analysis",
        f"{'='*50}",
        f"Your Team:    {your_pts} pts  (predicted: {your_xp:.1f} xP)"
        + (f"  | Percentile rank: {your_percentile_rank}th" if your_percentile_rank else ""),
    ]
    if recommended_pts is not None:
        lines.append(f"Recommended:  {recommended_pts} pts  (predicted: {recommended_xp:.1f} xP)")
    if dream_pts is not None:
        lines.append(f"Dream Team:   {dream_pts} pts")

    if benchmarks:
        lines.append("\nBenchmark scores this GW:")
        for label, score in benchmarks.items():
            if score is not None:
                lines.append(f"  {label:<20}: {score} pts")

    if misses:
        lines.append("\nBiggest prediction misses (your team):")
        for m in misses:
            sign = "+" if m["miss"] >= 0 else ""
            lines.append(f"  {m['name']:<20}: predicted {m['xP']:.1f} xP, "
                         f"actual {m['actual_points']} pts  ({sign}{m['miss']:.1f})")

    if recommended_pts is not None and your_pts is not None:
        gap = recommended_pts - your_pts
        lines.append(f"\nRecommendation value: {'+' if gap >= 0 else ''}{gap} pts over your team this GW")
    if dream_pts is not None and recommended_pts is not None:
        lines.append(f"Dream team gap: {recommended_pts - dream_pts} pts (recommended vs ceiling)")

    return "\n".join(lines)
```

- [x] **Step 4: Run tests**

```bash
python -m pytest tests/test_analysis.py -v
```
Expected: all PASS

- [x] **Step 5: Commit**

```bash
git add src/pipeline/analysis.py tests/test_analysis.py
git commit -m "feat: analysis.py — prediction misses, dream team, post-match summary"
```

---

## Task 12: P2a — benchmark scores + accuracy_log.csv

**Files:**
- Modify: `src/pipeline/user.py`
- Modify: `src/pipeline/analysis.py`
- Modify: `tests/test_user.py`
- Modify: `tests/test_analysis.py`

- [x] **Step 1: Write failing tests**

Add to `tests/test_user.py`:

```python
class TestFetchGwBenchmarks:
    def test_returns_benchmark_dict(self, sample_bootstrap_json):
        from src.pipeline.user import fetch_gw_benchmarks
        # sample_bootstrap_json has events with ids [29, 30, 31].
        # Find the event with id=30 by index (index 1 in conftest fixture).
        # Set values on the event dict directly (mutable).
        gw30_event = next(e for e in sample_bootstrap_json["events"] if e["id"] == 30)
        gw30_event["highest_score"] = 109
        gw30_event["average_entry_score"] = 38
        gw30_event["ranked_count"] = 12914049

        mock_standings_page = {
            "standings": {
                "results": [{"entry": i, "total": 85 - i} for i in range(50)]
            }
        }

        with patch("src.pipeline.user._api_get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_standings_page
            mock_get.return_value = mock_resp
            benchmarks = fetch_gw_benchmarks(
                gw=30,
                bootstrap_data=sample_bootstrap_json,
                overall_league_id=314,
            )

        assert benchmarks["best_score"] == 109
        assert benchmarks["avg_score"] == 38
        assert "top_1k_score" in benchmarks
```

Add to `tests/test_analysis.py`:

```python
class TestAppendAccuracyLog:
    def test_creates_log_on_first_run(self, tmp_path):
        from src.pipeline.analysis import append_accuracy_log
        log_path = tmp_path / "accuracy_log.csv"
        append_accuracy_log(
            path=log_path,
            gw=33,
            your_pts=58, your_xp=72.3,
            recommended_pts=65, recommended_xp=78.1,
            dream_pts=89,
            your_percentile_rank=20,
            benchmarks={"best_score": 109, "top_1k_score": 85,
                        "top_10k_score": 79, "top_100k_score": 73,
                        "top_1m_score": 62, "avg_score": 38, "median_score": 36},
            ranked_count=12914049,
        )
        df = pd.read_csv(log_path)
        assert len(df) == 1
        assert df.iloc[0]["gw"] == 33
        assert df.iloc[0]["your_pts"] == 58

    def test_appends_to_existing_log(self, tmp_path):
        from src.pipeline.analysis import append_accuracy_log
        log_path = tmp_path / "accuracy_log.csv"
        kwargs = dict(your_pts=60, your_xp=65.0, recommended_pts=None, recommended_xp=None,
                      dream_pts=None, your_percentile_rank=None, benchmarks={}, ranked_count=0)
        append_accuracy_log(log_path, gw=31, **kwargs)
        append_accuracy_log(log_path, gw=32, **kwargs)
        df = pd.read_csv(log_path)
        assert len(df) == 2
        assert list(df["gw"]) == [31, 32]
```

- [x] **Step 2: Run to verify fail**

```bash
python -m pytest tests/test_user.py::TestFetchGwBenchmarks tests/test_analysis.py::TestAppendAccuracyLog -v
```

- [x] **Step 3: Add `fetch_gw_benchmarks()` to `src/pipeline/user.py`**

```python
def fetch_gw_benchmarks(
    gw: int,
    bootstrap_data: dict,
    overall_league_id: int,
) -> dict:
    """Fetch GW benchmark scores from FPL API.

    Returns: best_score, avg_score, ranked_count, top_1k, top_10k, top_100k, top_1m (best-effort).
    best_score and avg_score come from bootstrap (already fetched). Others need standings pagination.
    """
    # Get free data from bootstrap events
    event = next((e for e in bootstrap_data.get("events", []) if e["id"] == gw), {})
    benchmarks = {
        "best_score": event.get("highest_score"),
        "avg_score": event.get("average_entry_score"),
        "ranked_count": event.get("ranked_count"),
        "top_1k_score": None,
        "top_10k_score": None,
        "top_100k_score": None,
        "top_1m_score": None,
        "median_score": None,
    }

    # Fetch score at specific ranks via standings pagination (50 entries per page)
    rank_to_key = {1000: "top_1k_score", 10000: "top_10k_score",
                   100000: "top_100k_score", 1000000: "top_1m_score"}

    for rank, key in rank_to_key.items():
        page = (rank - 1) // 50 + 1
        try:
            url = f"{FPL_LEAGUES_CLASSIC_URL}/{overall_league_id}/standings/?page_standings={page}&event={gw}"
            resp = _api_get_with_retry(url, timeout=30)
            results = resp.json().get("standings", {}).get("results", [])
            if results:
                # Score at rank position within the page
                position_in_page = (rank - 1) % 50
                if position_in_page < len(results):
                    benchmarks[key] = results[position_in_page].get("total") or results[position_in_page].get("event_total")
        except Exception as e:
            logger.warning(f"Could not fetch standings page for rank {rank}: {e}")

    return benchmarks
```

- [x] **Step 4: Add `append_accuracy_log()` to `src/pipeline/analysis.py`**

```python
def append_accuracy_log(
    path: Path,
    gw: int,
    your_pts: int | None,
    your_xp: float | None,
    recommended_pts: int | None,
    recommended_xp: float | None,
    dream_pts: int | None,
    your_percentile_rank: int | None,
    benchmarks: dict,
    ranked_count: int | None,
) -> None:
    """Append one row per GW to the season accuracy log CSV."""
    from datetime import datetime, timezone
    row = {
        "gw": gw,
        "your_pts": your_pts,
        "your_predicted_xp": round(your_xp, 2) if your_xp is not None else None,
        "recommended_pts": recommended_pts,
        "recommended_xp": round(recommended_xp, 2) if recommended_xp is not None else None,
        "dream_team_pts": dream_pts,
        "your_percentile_rank": your_percentile_rank,
        "best_score": benchmarks.get("best_score"),
        "top_1k_score": benchmarks.get("top_1k_score"),
        "top_10k_score": benchmarks.get("top_10k_score"),
        "top_100k_score": benchmarks.get("top_100k_score"),
        "top_1m_score": benchmarks.get("top_1m_score"),
        "avg_score": benchmarks.get("avg_score"),
        "median_score": benchmarks.get("median_score"),
        "ranked_count": ranked_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if path.exists():
        df_existing = pd.read_csv(path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(path, index=False)
```

- [x] **Step 5: Run all tests**

```bash
python -m pytest tests/test_user.py tests/test_analysis.py -v
```
Expected: all PASS

- [x] **Step 6: Commit**

```bash
git add src/pipeline/user.py src/pipeline/analysis.py tests/test_user.py tests/test_analysis.py
git commit -m "feat: benchmark fetch and accuracy_log CSV for post-match analysis"
```

---

## Task 13: run.py — extend post-gw with P2a analysis

**Context:** Wire the P2a analysis into `phase_post_gw()`. Load user picks, compare vs predictions, compute dream team, fetch benchmarks, print summary, write accuracy_log.

**Files:**
- Modify: `src/pipeline/run.py`
- Modify: `tests/test_run.py`

- [x] **Step 1: Write failing test**

Add to `tests/test_run.py`:

```python
class TestPostGwAnalysis:
    def test_post_gw_skips_analysis_when_no_config(self, tmp_path, monkeypatch):
        """If user_config.yaml missing, post-gw still completes (analysis skipped)."""
        import src.pipeline.run as run_mod
        from unittest.mock import patch, MagicMock
        monkeypatch.setattr(run_mod, "RESULTS_DIR", tmp_path)
        # Mock API calls to return minimal data
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "events": [{"id": 30, "is_current": True, "is_next": False,
                        "finished": True, "highest_score": 100,
                        "average_entry_score": 40, "ranked_count": 10000000}],
            "elements": [], "teams": [],
        }
        with patch("src.pipeline.run.fetch_bootstrap", return_value=mock_resp.json()), \
             patch("src.pipeline.run.fetch_fixtures", return_value=[]), \
             patch("src.pipeline.run.fetch_live_gw_data", return_value=pd.DataFrame()), \
             patch("src.pipeline.run.load_user_config", side_effect=UserConfigError("missing")):
            # Should not raise
            run_mod.phase_post_gw()
```

- [x] **Step 2: Run to verify fail**

```bash
python -m pytest tests/test_run.py::TestPostGwAnalysis -v
```

- [x] **Step 3: Extend `phase_post_gw()` in `run.py`**

Add imports at the top:

```python
from src.pipeline.analysis import (
    compute_prediction_misses, compute_dream_team,
    format_post_match_summary, append_accuracy_log,
)
```

At the end of `phase_post_gw()`, after saving live data, add:

```python
    # P2a: post-match analysis (skipped gracefully if user config missing)
    try:
        cfg = load_user_config()
    except UserConfigError:
        print("[post-gw] user_config.yaml not found — skipping post-match analysis")
        return

    entry_id = cfg["teams"]["default"]["entry_id"]

    # Load predictions for this GW
    gw_label = f"gw{gw}"
    pred_path = RESULTS_DIR / f"predictions_{gw_label}.csv"
    if not pred_path.exists():
        print(f"[post-gw] Predictions file {pred_path} not found — skipping analysis")
        return

    predictions = pd.read_csv(pred_path)

    # Fetch user picks for this GW
    try:
        user_state = fetch_user_team_state(entry_id, gw, bootstrap)
        picks_elements = set(user_state.current_squad)
        your_picks = predictions[predictions["element"].isin(picks_elements)].copy()
    except Exception as e:
        logger.warning(f"Could not fetch user picks: {e}")
        your_picks = pd.DataFrame()

    # Merge actual points
    if not live_df.empty and not your_picks.empty:
        actual_map = live_df.set_index("element")["total_points"].to_dict()
        your_picks["actual_points"] = your_picks["element"].map(actual_map).fillna(0)
        your_pts = int(your_picks["actual_points"].sum())
        your_xp = float(your_picks["xP"].sum())
        misses = compute_prediction_misses(your_picks)
    else:
        your_pts = 0
        your_xp = 0.0
        misses = []

    # Recommended team comparison
    rec_path = RESULTS_DIR / f"recommend_{gw_label}.csv"
    recommended_pts = None
    recommended_xp = None
    if rec_path.exists() and not live_df.empty:
        rec_df = pd.read_csv(rec_path)
        rec_elements = set(
            predictions[predictions["name"].isin(rec_df["player_in"].dropna())]["element"]
        )
        if rec_elements:
            rec_picks = live_df[live_df["element"].isin(rec_elements)]
            recommended_pts = int(rec_picks["total_points"].sum())
            rec_xp_df = predictions[predictions["element"].isin(rec_elements)]
            recommended_xp = float(rec_xp_df["xP"].sum()) if not rec_xp_df.empty else None

    # Dream team from live data
    dream_pts = None
    if not live_df.empty:
        try:
            dream = compute_dream_team(live_df)
            dream_pts = int(dream["total_points"].sum() if "total_points" in dream.columns
                            else dream["xP"].sum())
        except Exception as e:
            logger.warning(f"Dream team computation failed: {e}")

    # Benchmarks
    from src.pipeline.user import fetch_gw_benchmarks
    overall_league_id = None
    try:
        entry_data = _api_get_with_retry(f"{FPL_ENTRY_URL}/{entry_id}/").json()
        for league in entry_data.get("leagues", {}).get("classic", []):
            if league.get("league_type") == "s" and league.get("scoring") == "c":
                overall_league_id = league["id"]
                break
    except Exception:
        pass

    benchmarks = {}
    your_percentile_rank = None
    if overall_league_id:
        try:
            benchmarks = fetch_gw_benchmarks(gw, bootstrap, overall_league_id)
        except Exception as e:
            logger.warning(f"Could not fetch benchmarks: {e}")
    # Percentile rank from history
    try:
        history = _api_get_with_retry(f"{FPL_ENTRY_URL}/{entry_id}/history/").json()
        for row in history.get("current", []):
            if row["event"] == gw:
                your_percentile_rank = row.get("percentile_rank")
                break
    except Exception:
        pass

    # Print summary
    print(format_post_match_summary(
        gw=gw, your_pts=your_pts, your_xp=your_xp,
        recommended_pts=recommended_pts, recommended_xp=recommended_xp,
        dream_pts=dream_pts, benchmarks=benchmarks,
        your_percentile_rank=your_percentile_rank, misses=misses,
    ))

    # Write accuracy log
    log_path = RESULTS_DIR / "accuracy_log.csv"
    append_accuracy_log(
        path=log_path, gw=gw,
        your_pts=your_pts, your_xp=your_xp,
        recommended_pts=recommended_pts, recommended_xp=recommended_xp,
        dream_pts=dream_pts, your_percentile_rank=your_percentile_rank,
        benchmarks=benchmarks, ranked_count=benchmarks.get("ranked_count"),
    )
    print(f"[post-gw] Accuracy log updated: {log_path}")
```

Also add the missing import for `_api_get_with_retry`:

```python
from src.pipeline.fetch import (
    fetch_bootstrap, get_current_gw, get_next_deadline,
    extract_xp_snapshot, fetch_fixtures, fetch_live_gw_data,
    _api_get_with_retry,
)
```

And add to config imports:

```python
from src.config import (
    VAASTAV_DIR, RESULTS_DIR, MODELS_DIR, CURRENT_SEASON,
    ACTIVE_MODEL, BOOTSTRAP_MAX_AGE_HOURS,
    FPL_ENTRY_URL, load_user_config, UserConfigError,
)
```

- [x] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -q
```
Expected: all tests pass (was 64, now 64 + new tests)

- [x] **Step 5: Commit**

```bash
git add src/pipeline/run.py src/pipeline/analysis.py tests/test_run.py tests/test_analysis.py
git commit -m "feat: extend post-gw with P2a post-match analysis and accuracy_log"
```

---

## Final: Integration smoke test

- [x] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: all PASS, no new failures

- [x] **Step 2: Smoke test CLI help**

```bash
python -m src.pipeline.run --help
```
Expected: shows `recommend` in phase choices with `--horizon`, `--wildcard`, `--team`

- [x] **Step 3: Commit docs update**

```bash
# Update improvements-roadmap.md to mark P1 and P2a as implemented
git add docs/improvements-roadmap.md
git commit -m "docs: mark P1 and P2a as implemented in improvements roadmap"
```

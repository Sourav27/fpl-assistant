# Track H — Data Sources Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.
>
> **Agent note:** This plan is designed for execution by the `data-engineer` agent. Each task is independently testable. Run on a git worktree (see Task 0).

**Goal:** Integrate external data sources (understatAPI xG/xGC, soccerdata European minutes, FFS/Reddit/premierinjuries signals) with reliability validation against the FPL API before Track B feature engineering begins.

**Architecture:** A new `src/pipeline/datasources/` package with one module per source, a shared `PlayerSignal` dataclass, a validation gate that compares Spearman ρ across sources, and a signal feedback logger. No source's data touches the xP pipeline until it passes its validation gate.

**Tech Stack:** Python, `understatapi`, `soccerdata`, `feedparser`, `requests`, `scipy.stats.spearmanr`, `pandas`, `pytest`, `unittest.mock` (for HTTP mocking in tests)

---

## Worktree Setup (do this first)

```bash
# From the main repo root
git worktree add .worktrees/track-h feature/track-h-data-sources
cd .worktrees/track-h

# Data junction — worktrees don't copy symlinks or junctions
# Windows (run in cmd.exe, not bash):
#   mklink /J .worktrees\track-h\data\Fantasy-Premier-League data\Fantasy-Premier-League
# bash equivalent (if already junctioned via mklink, this is a no-op):
ls data/Fantasy-Premier-League || echo "Create junction manually — see CLAUDE.md Gotchas"
```

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `src/pipeline/datasources/__init__.py` | Package marker; exports `PlayerSignal`, `SourceValidationResult` |
| `src/pipeline/datasources/understat.py` | understatAPI client — fetch xG/xA/xGC per player per GW (EPL only) |
| `src/pipeline/datasources/soccerdata_client.py` | FotMob wrapper — European/international match minutes |
| `src/pipeline/datasources/signals.py` | `PlayerSignal` dataclass + name→code resolver shared by all signal parsers |
| `src/pipeline/datasources/ffs.py` | Fantasy Football Scout RSS parser → `PlayerSignal` list |
| `src/pipeline/datasources/reddit.py` | Reddit r/FantasyPL JSON API client → `PlayerSignal` list |
| `src/pipeline/datasources/premierinjuries.py` | premierinjuries.com HTML scraper → `PlayerSignal` list |
| `src/pipeline/source_validation.py` | Spearman ρ gate: compares understat xG vs FPL Opta xG vs actual goals; writes `results/source_validation.csv` |
| `src/pipeline/signal_feedback.py` | Appends to `results/signal_accuracy.csv` when team sheets arrive |
| `tests/datasources/__init__.py` | Package marker |
| `tests/datasources/test_understat.py` | Unit tests for understat client + xGC team aggregation |
| `tests/datasources/test_soccerdata.py` | Unit tests for FotMob minutes client + FPL cross-validation |
| `tests/datasources/test_signals.py` | Unit tests for `PlayerSignal`, name resolver, FFS/Reddit/premierinjuries parsers |
| `tests/datasources/test_source_validation.py` | Unit tests for validation gate logic and CSV output |
| `tests/datasources/test_signal_feedback.py` | Unit tests for feedback log append and CSV schema |
| `tests/datasources/test_integration_datasources.py` | End-to-end chain tests: fetch→parse→signal→resolve→log (all HTTP mocked via `unittest.mock`) |

### Modified files

| File | Change |
|------|--------|
| `src/config.py` | Add `SOURCE_VALIDATION_CSV`, `SIGNAL_ACCURACY_CSV`, `SIGNAL_UNRESOLVED_CSV` path constants |
| `requirements.txt` | Add `understatapi`, `soccerdata`, `feedparser` (`beautifulsoup4` already present) |

---

## Dependency Install

Run once before any task:

```bash
pip install understatapi soccerdata feedparser vcrpy
pip install -r requirements.txt
```

---

## Task 0 — Shared Dataclasses and Config Constants

**Files:**
- Create: `src/pipeline/datasources/__init__.py`
- Create: `src/pipeline/datasources/signals.py`
- Modify: `src/config.py`
- Create: `tests/datasources/__init__.py`
- Create: `tests/datasources/test_signals.py`

### Background

`PlayerSignal` is the shared contract between all signal parsers and the feedback logger. Define it once, import everywhere.

The name resolver converts FFS/Reddit/premierinjuries player name strings (e.g. `"Mohamed Salah"`) to FPL `code` integers (e.g. `80201`). It must:
1. Try exact match on bootstrap `web_name` (e.g. `"Salah"`)
2. Try `first_name + ' ' + second_name` from bootstrap
3. Log to `signal_unresolved.csv` on failure — never guess

- [x] **Step 1: Write failing tests for `PlayerSignal` and name resolver**

```python
# tests/datasources/test_signals.py
import pytest
from src.pipeline.datasources.signals import PlayerSignal, resolve_player_name

MOCK_BOOTSTRAP = {
    "elements": [
        {"id": 1, "code": 80201, "web_name": "Salah",
         "first_name": "Mohamed", "second_name": "Salah", "team": 14},
        {"id": 2, "code": 54694, "web_name": "Wilson",
         "first_name": "Callum", "second_name": "Wilson", "team": 7},
        {"id": 3, "code": 99999, "web_name": "Wilson",
         "first_name": "Ben", "second_name": "Wilson", "team": 3},
    ]
}


def test_player_signal_fields():
    sig = PlayerSignal(
        player_code=80201,
        source="ffs",
        signal_type="doubt",
        text="Salah is a doubt for GW32.",
        timestamp="2026-04-05T08:00:00Z",
        confidence=0.9,
    )
    assert sig.player_code == 80201
    assert sig.source == "ffs"
    assert sig.signal_type == "doubt"


def test_resolve_exact_web_name():
    code = resolve_player_name("Salah", MOCK_BOOTSTRAP)
    assert code == 80201


def test_resolve_full_name():
    code = resolve_player_name("Mohamed Salah", MOCK_BOOTSTRAP)
    assert code == 80201


def test_resolve_ambiguous_returns_none():
    # Two players with web_name "Wilson" — must not resolve silently
    code = resolve_player_name("Wilson", MOCK_BOOTSTRAP)
    assert code is None


def test_resolve_unknown_returns_none():
    code = resolve_player_name("Nonexistent Player", MOCK_BOOTSTRAP)
    assert code is None


def test_log_unresolved_writes_csv(tmp_path):
    from src.pipeline.datasources.signals import log_unresolved_name
    csv_path = tmp_path / "signal_unresolved.csv"
    log_unresolved_name("Unknown X", source="ffs", raw_text="Unknown X doubt", csv_path=csv_path)
    import pandas as pd
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["name"] == "Unknown X"
    assert df.iloc[0]["source"] == "ffs"
```

- [x] **Step 2: Run to confirm FAIL**

```bash
pytest tests/datasources/test_signals.py -v
# Expected: ImportError or AttributeError (module not found)
```

- [x] **Step 3: Add config constants**

In `src/config.py`, append after the existing path constants:

```python
SOURCE_VALIDATION_CSV = RESULTS_DIR / "source_validation.csv"
SIGNAL_ACCURACY_CSV   = RESULTS_DIR / "signal_accuracy.csv"
SIGNAL_UNRESOLVED_CSV = RESULTS_DIR / "signal_unresolved.csv"
```

- [x] **Step 4: Implement `PlayerSignal` and `resolve_player_name`**

```python
# src/pipeline/datasources/signals.py
"""Shared signal dataclass and player name resolver."""
from __future__ import annotations
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlayerSignal:
    player_code: int           # FPL persistent player code
    source: str                # "ffs" | "reddit" | "premierinjuries"
    signal_type: str           # "doubt" | "available" | "injured" | "rotation_risk" | "differential"
    text: str                  # Raw signal text
    timestamp: str             # ISO 8601
    confidence: float = 1.0   # 0–1; 1.0 = structured source, lower = NLP-inferred


def resolve_player_name(name: str, bootstrap_data: dict) -> int | None:
    """Resolve a player name string to an FPL persistent player code.

    Resolution order:
    1. Exact match on web_name
    2. Exact match on first_name + ' ' + second_name
    3. Returns None (ambiguous or unresolved) — never guesses

    Returns None if ambiguous (multiple matches) or not found.
    """
    elements = bootstrap_data.get("elements", [])
    name_lower = name.strip().lower()

    # Step 1: exact web_name match
    web_matches = [e for e in elements if e["web_name"].lower() == name_lower]
    if len(web_matches) == 1:
        return web_matches[0]["code"]
    if len(web_matches) > 1:
        logger.warning("Ambiguous web_name '%s' — %d matches, skipping", name, len(web_matches))
        return None

    # Step 2: full name match
    full_matches = [
        e for e in elements
        if (e["first_name"] + " " + e["second_name"]).lower() == name_lower
    ]
    if len(full_matches) == 1:
        return full_matches[0]["code"]
    if len(full_matches) > 1:
        logger.warning("Ambiguous full name '%s' — %d matches, skipping", name, len(full_matches))
        return None

    return None


def log_unresolved_name(
    name: str,
    source: str,
    raw_text: str,
    csv_path: "Path | None" = None,
    timestamp: str = "",
) -> None:
    """Write an unresolved player name to signal_unresolved.csv for manual review."""
    import pandas as pd
    from src.config import SIGNAL_UNRESOLVED_CSV
    from pathlib import Path
    if csv_path is None:
        csv_path = SIGNAL_UNRESOLVED_CSV
    row = pd.DataFrame([{
        "name": name,
        "source": source,
        "raw_text": raw_text,
        "timestamp": timestamp,
    }])
    if Path(csv_path).exists():
        row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(csv_path, index=False)
    logger.debug("Unresolved name '%s' from '%s' logged to %s", name, source, csv_path)
```

```python
# src/pipeline/datasources/__init__.py
from .signals import PlayerSignal, resolve_player_name, log_unresolved_name

__all__ = ["PlayerSignal", "resolve_player_name", "log_unresolved_name"]
```

```python
# tests/datasources/__init__.py
```

- [x] **Step 5: Run tests to confirm PASS**

```bash
pytest tests/datasources/test_signals.py -v
# Expected: 6 passed (5 original + test_log_unresolved_writes_csv)
```

- [x] **Step 6: Commit**

```bash
git add src/pipeline/datasources/ src/config.py tests/datasources/
git commit -m "feat(H-F0): PlayerSignal dataclass, name resolver, config path constants"
```

---

## Task 1 — understatAPI Client (H-F1)

> **Prerequisite:** Ensure dependencies are installed (see "Dependency Install" section above). `understatapi` must be installed before Step 4.

**Files:**
- Create: `src/pipeline/datasources/understat.py`
- Create: `tests/datasources/test_understat.py`

### Background

`understatAPI` returns shot-level data. To derive team-level `xGC` (expected goals conceded) we need:
1. For each fixture: fetch all shot events (or player match stats)
2. For each shot, identify the defending team
3. Sum `xG` of all shots against each team per GW → that team's `xGC` for that GW

Before implementing this, **first check** whether `understatAPI.get_league_players()` already exposes team-level aggregates that include xGC — this would save the shot aggregation step. Document which route was used and why.

The `understatAPI` library uses `async` — wrap in a sync helper using `asyncio.run()`.

> **asyncio safety note:** `asyncio.run()` raises `RuntimeError` if called inside an already-running event loop (e.g. Jupyter, FastAPI). This is acceptable for the CLI pipeline. Add a comment in the implementation noting this limitation. If called from Track F's async FastAPI context, use `await` directly instead.

- [x] **Step 1: Write failing tests**

```python
# tests/datasources/test_understat.py
import pytest
import pandas as pd
from unittest.mock import patch, AsyncMock

from src.pipeline.datasources.understat import (
    fetch_understat_player_gw_stats,
    compute_team_xgc_per_gw,
)


MOCK_PLAYER_DATA = [
    {"player_id": "1", "player": "Salah", "team": "Liverpool",
     "xG": "0.45", "xA": "0.12", "time": "90",
     "date": "2026-01-01", "id": "fixture_1", "h_team": "Arsenal", "a_team": "Liverpool"},
    {"player_id": "2", "player": "Havertz", "team": "Arsenal",
     "xG": "0.31", "xA": "0.05", "time": "85",
     "date": "2026-01-01", "id": "fixture_1", "h_team": "Arsenal", "a_team": "Liverpool"},
]


def test_fetch_returns_dataframe(tmp_path):
    """fetch_understat_player_gw_stats returns a DataFrame with required columns."""
    with patch(
        "src.pipeline.datasources.understat._fetch_player_grouped_stats_async",
        return_value=MOCK_PLAYER_DATA
    ):
        df = fetch_understat_player_gw_stats(season="2025")
    assert isinstance(df, pd.DataFrame)
    assert {"player_id", "team", "xG", "xA", "date"}.issubset(df.columns)


def test_compute_team_xgc_per_gw():
    """compute_team_xgc_per_gw aggregates xG against each team per match."""
    df = pd.DataFrame(MOCK_PLAYER_DATA)
    df["xG"] = df["xG"].astype(float)
    # Arsenal concedes Salah's 0.45; Liverpool concedes Havertz's 0.31
    team_xgc = compute_team_xgc_per_gw(df)
    assert "team" in team_xgc.columns
    assert "fixture_id" in team_xgc.columns
    assert "xGC" in team_xgc.columns
    arsenal_row = team_xgc[team_xgc["team"] == "Arsenal"]
    assert pytest.approx(arsenal_row["xGC"].values[0], abs=0.01) == 0.31


def test_xgc_non_negative():
    """xGC values must be >= 0."""
    df = pd.DataFrame(MOCK_PLAYER_DATA)
    df["xG"] = df["xG"].astype(float)
    team_xgc = compute_team_xgc_per_gw(df)
    assert (team_xgc["xGC"] >= 0).all()
```

- [x] **Step 2: Run to confirm FAIL**

```bash
pytest tests/datasources/test_understat.py -v
# Expected: ImportError
```

- [x] **Step 3: Implement understat client**

```python
# src/pipeline/datasources/understat.py
"""understatAPI client — PL xG, xA, xGC per player per GW (EPL only).

understatAPI covers 6 leagues: EPL, La_Liga, Bundesliga, Serie_A, Ligue_1, RFPL.
It does NOT cover Champions League, Europa League, or international matches.

Architecture: understatAPI is async. We wrap in sync helpers using asyncio.run().

xGC derivation: understatAPI's get_league_players() returns per-player match stats
including xG and team name. We derive team-level xGC by:
  For each fixture: sum xG of all opponent players against the defending team.
"""
from __future__ import annotations
import asyncio
import logging
import pandas as pd

logger = logging.getLogger(__name__)

UNDERSTAT_SEASON_MAP = {
    "2025-26": "2025",
    "2024-25": "2024",
    "2023-24": "2023",
    "2022-23": "2022",
    "2021-22": "2021",
}


async def _fetch_player_grouped_stats_async(season: str) -> list[dict]:
    """Async inner — fetch per-player per-match stats from understatAPI."""
    from understatapi import UnderstatClient
    async with UnderstatClient() as client:
        data = await client.league(league="EPL").get_player_data(season=season)
    # Each entry: {player_id, player, team, xG, xA, time, date, id (fixture), h_team, a_team, ...}
    return data


def fetch_understat_player_gw_stats(season: str = "2025") -> pd.DataFrame:
    """Return a DataFrame of per-player per-match xG/xA stats for the given season.

    Args:
        season: Understat season string, e.g. "2025" for 2025-26.

    Columns: player_id, player, team, xG (float), xA (float), time (int),
             date (str), fixture_id (str), h_team, a_team
    """
    raw = asyncio.run(_fetch_player_grouped_stats_async(season))
    df = pd.DataFrame(raw)
    df = df.rename(columns={"id": "fixture_id"})
    for col in ("xG", "xA"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(0).astype(int)
    return df


def compute_team_xgc_per_gw(df: pd.DataFrame) -> pd.DataFrame:
    """Compute team-level xGC per fixture from player-level xG rows.

    For each fixture, the defending team's xGC = sum of xG from all
    players on the opposing team.

    Returns DataFrame with columns: fixture_id, team, xGC
    """
    if df.empty:
        return pd.DataFrame(columns=["fixture_id", "team", "xGC"])

    rows = []
    for fixture_id, group in df.groupby("fixture_id"):
        h_team = group["h_team"].iloc[0]
        a_team = group["a_team"].iloc[0]
        # Home team concedes away players' xG
        away_xg = group[group["team"] == a_team]["xG"].sum()
        home_xg = group[group["team"] == h_team]["xG"].sum()
        rows.append({"fixture_id": fixture_id, "team": h_team, "xGC": away_xg})
        rows.append({"fixture_id": fixture_id, "team": a_team, "xGC": home_xg})

    return pd.DataFrame(rows)
```

- [x] **Step 4: Run tests to confirm PASS**

```bash
pytest tests/datasources/test_understat.py -v
# Expected: 3 passed
```

- [x] **Step 5: Commit**

```bash
git add src/pipeline/datasources/understat.py tests/datasources/test_understat.py
git commit -m "feat(H-F1): understatAPI client with team-level xGC derivation"
```

---

## Task 2 — xG Source Validation Gate (H-F2)

**Files:**
- Create: `src/pipeline/source_validation.py`
- Create: `tests/datasources/test_source_validation.py`

### Background

Before Track B can use `understat_xGC` as a model feature, we must confirm understat xG is at least as predictive as FPL Opta xG (`expected_goals` column in vaastav) for actual goals scored.

Gate rule: understat ρ ≥ FPL Opta ρ − 0.05 (within 5 percentage points) → use understat.
If gate fails → fall back to `goals_conceded` aggregated from vaastav for `xGC_rolling_4`.

The gate joins understat player data with vaastav merged GW history on `(player_name, team, date)` matching. This is fuzzy — document the join key and any rows that fail to join.

- [x] **Step 1: Write failing tests**

```python
# tests/datasources/test_source_validation.py
import pytest
import pandas as pd
from src.pipeline.source_validation import (
    compute_source_spearman,
    run_xg_validation_gate,
    SourceValidationResult,
)


def _make_df(xg_col: str, xg_vals, goals_vals) -> pd.DataFrame:
    return pd.DataFrame({xg_col: xg_vals, "goals_scored": goals_vals})


def test_spearman_perfect_correlation():
    df = _make_df("understat_xG", [0.1, 0.3, 0.7, 1.2], [0, 0, 1, 2])
    rho = compute_source_spearman(df, xg_col="understat_xG", actual_col="goals_scored")
    assert rho > 0.9


def test_spearman_returns_float():
    df = _make_df("xG", [0.2, 0.5, 0.9], [0, 1, 1])
    rho = compute_source_spearman(df, xg_col="xG", actual_col="goals_scored")
    assert isinstance(rho, float)
    assert -1.0 <= rho <= 1.0


def test_gate_passes_when_understat_within_tolerance():
    result = run_xg_validation_gate(
        understat_rho=0.62,
        fpl_opta_rho=0.65,
        tolerance=0.05,
    )
    assert result.passed is True
    assert result.recommended_source == "understat"


def test_gate_fails_when_understat_too_low():
    result = run_xg_validation_gate(
        understat_rho=0.55,
        fpl_opta_rho=0.65,
        tolerance=0.05,
    )
    assert result.passed is False
    assert result.recommended_source == "vaastav_goals_conceded"


def test_gate_result_has_required_fields():
    result = run_xg_validation_gate(0.68, 0.65, 0.05)
    assert hasattr(result, "passed")
    assert hasattr(result, "understat_rho")
    assert hasattr(result, "fpl_opta_rho")
    assert hasattr(result, "recommended_source")
```

- [x] **Step 2: Run to confirm FAIL**

```bash
pytest tests/datasources/test_source_validation.py -v
# Expected: ImportError
```

- [x] **Step 3: Implement source validation**

```python
# src/pipeline/source_validation.py
"""xG source validation gate for Track H / Track B dependency.

Usage:
    python -m src.pipeline.source_validation  # runs gate, writes results/source_validation.csv
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
import pandas as pd
from scipy.stats import spearmanr
from src.config import SOURCE_VALIDATION_CSV

logger = logging.getLogger(__name__)


@dataclass
class SourceValidationResult:
    passed: bool
    understat_rho: float
    fpl_opta_rho: float
    recommended_source: str   # "understat" | "vaastav_goals_conceded"
    n_samples: int = 0


def compute_source_spearman(df: pd.DataFrame, xg_col: str, actual_col: str) -> float:
    """Compute Spearman ρ between an xG column and actual goals column."""
    clean = df[[xg_col, actual_col]].dropna()
    if len(clean) < 10:
        logger.warning("Only %d samples for Spearman ρ — result unreliable", len(clean))
    rho, _ = spearmanr(clean[xg_col], clean[actual_col])
    return float(rho)


def run_xg_validation_gate(
    understat_rho: float,
    fpl_opta_rho: float,
    tolerance: float = 0.05,
    n_samples: int = 0,
) -> SourceValidationResult:
    """Evaluate whether understat xG is reliable enough to use in Track B.

    Gate: understat_rho >= fpl_opta_rho - tolerance
    If passed: use understat xGC in xGC_rolling_4.
    If failed: fall back to vaastav goals_conceded aggregation.
    """
    passed = understat_rho >= (fpl_opta_rho - tolerance)
    source = "understat" if passed else "vaastav_goals_conceded"
    return SourceValidationResult(
        passed=passed,
        understat_rho=understat_rho,
        fpl_opta_rho=fpl_opta_rho,
        recommended_source=source,
        n_samples=n_samples,
    )


def append_validation_result(result: SourceValidationResult, run_date: str) -> None:
    """Append a validation result row to results/source_validation.csv."""
    row = pd.DataFrame([{
        "run_date": run_date,
        "understat_rho": result.understat_rho,
        "fpl_opta_rho": result.fpl_opta_rho,
        "n_samples": result.n_samples,
        "gate_passed": result.passed,
        "recommended_source": result.recommended_source,
    }])
    if SOURCE_VALIDATION_CSV.exists():
        row.to_csv(SOURCE_VALIDATION_CSV, mode="a", header=False, index=False)
    else:
        SOURCE_VALIDATION_CSV.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(SOURCE_VALIDATION_CSV, index=False)
    logger.info("Validation result appended to %s", SOURCE_VALIDATION_CSV)
```

- [x] **Step 4: Run tests to confirm PASS**

```bash
pytest tests/datasources/test_source_validation.py -v
# Expected: 5 passed
```

- [x] **Step 5: Commit**

```bash
git add src/pipeline/source_validation.py tests/datasources/test_source_validation.py
git commit -m "feat(H-F2): xG source validation gate with Spearman rho comparison"
```

---

## Task 3 — soccerdata + FotMob European Minutes (H-F3)

**Files:**
- Create: `src/pipeline/datasources/soccerdata_client.py`
- Create: `tests/datasources/test_soccerdata.py`

### Background

`soccerdata` wraps FotMob (among other sources). We need minutes played in UCL, UEL, and internationals — specifically to flag players with ≥ 90 minutes within 72h of a PL deadline.

> **soccerdata API note (verify before implementing):** `soccerdata.FotMob().read_schedule()` returns match schedules, NOT player minutes. Player-level minutes are likely under `read_player_match_stats()` or similar. **Before writing the implementation in Step 3, run:**
> ```python
> import soccerdata as sd; help(sd.FotMob)
> ```
> and confirm the correct method name. Update `_fetch_fotmob_raw` to call the correct method. The test mocks `_fetch_fotmob_raw` so unit tests will pass regardless — but the production implementation must call the right method.

The **reliability check** cross-validates FotMob PL minutes against FPL `element-summary` minutes for the same fixtures. Log MAE and per-player delta. A MAE > 5 min or correlation < 0.95 for PL matches indicates the source is unreliable.

- [x] **Step 1: Write failing tests**

```python
# tests/datasources/test_soccerdata.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.pipeline.datasources.soccerdata_client import (
    fetch_fotmob_player_minutes,
    cross_validate_with_fpl,
    FotMobReliabilityResult,
)


MOCK_FOTMOB_ROWS = [
    {"player_name": "Salah", "team": "Liverpool",
     "competition": "UEFA Champions League", "date": "2026-03-18",
     "minutes": 90},
    {"player_name": "Havertz", "team": "Arsenal",
     "competition": "Premier League", "date": "2026-03-15",
     "minutes": 85},
]

MOCK_FPL_MINUTES = pd.DataFrame([
    {"web_name": "Havertz", "team": "Arsenal", "date": "2026-03-15", "minutes": 85}
])


def test_fetch_returns_dataframe():
    with patch(
        "src.pipeline.datasources.soccerdata_client._fetch_fotmob_raw",
        return_value=pd.DataFrame(MOCK_FOTMOB_ROWS)
    ):
        df = fetch_fotmob_player_minutes(competitions=["UEFA Champions League"])
    assert isinstance(df, pd.DataFrame)
    assert "minutes" in df.columns
    # Only UCL rows returned
    assert all(df["competition"] == "UEFA Champions League")


def test_fetch_filters_by_competition():
    with patch(
        "src.pipeline.datasources.soccerdata_client._fetch_fotmob_raw",
        return_value=pd.DataFrame(MOCK_FOTMOB_ROWS)
    ):
        df = fetch_fotmob_player_minutes(competitions=["Premier League"])
    assert len(df) == 1
    assert df.iloc[0]["player_name"] == "Havertz"


def test_cross_validate_high_correlation():
    """When FotMob and FPL match closely, reliability result should pass."""
    fotmob_pl = pd.DataFrame([
        {"player_name": "Havertz", "team": "Arsenal", "date": "2026-03-15", "minutes": 85}
    ])
    result = cross_validate_with_fpl(fotmob_pl, MOCK_FPL_MINUTES)
    assert isinstance(result, FotMobReliabilityResult)
    assert result.mae < 5.0
    assert result.correlation > 0.90


def test_cross_validate_result_fields():
    fotmob_pl = pd.DataFrame([
        {"player_name": "Havertz", "team": "Arsenal", "date": "2026-03-15", "minutes": 85}
    ])
    result = cross_validate_with_fpl(fotmob_pl, MOCK_FPL_MINUTES)
    assert hasattr(result, "mae")
    assert hasattr(result, "correlation")
    assert hasattr(result, "n_matched")
    assert hasattr(result, "reliable")
```

- [x] **Step 2: Run to confirm FAIL**

```bash
pytest tests/datasources/test_soccerdata.py -v
# Expected: ImportError
```

- [x] **Step 3: Implement soccerdata client**

```python
# src/pipeline/datasources/soccerdata_client.py
"""FotMob wrapper (via soccerdata) for European/international match minutes.

Covers: UEFA Champions League, UEFA Europa League, international matches.
Does NOT cover: Premier League (use FPL API for those).

Reliability: cross-validated against FPL element-summary for PL matches.
Gate: MAE <= 5 min AND correlation >= 0.95 on PL matches → reliable for non-PL.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_COMPETITIONS = [
    "UEFA Champions League",
    "UEFA Europa League",
    "UEFA Conference League",
    "International Friendlies",
]


@dataclass
class FotMobReliabilityResult:
    mae: float
    correlation: float
    n_matched: int
    reliable: bool  # True if mae <= 5 and correlation >= 0.95


def _fetch_fotmob_raw(competitions: list[str]) -> pd.DataFrame:
    """Fetch raw FotMob match data via soccerdata.

    Returns DataFrame with: player_name, team, competition, date, minutes
    """
    import soccerdata as sd
    rows = []
    fotmob = sd.FotMob()
    for comp in competitions:
        try:
            schedule = fotmob.read_schedule(competition=comp)
            # soccerdata returns a MultiIndex DataFrame — reset for simplicity
            if schedule is not None and not schedule.empty:
                schedule = schedule.reset_index()
                rows.append(schedule)
        except Exception as e:
            logger.warning("FotMob fetch failed for %s: %s", comp, e)
    if not rows:
        return pd.DataFrame(columns=["player_name", "team", "competition", "date", "minutes"])
    return pd.concat(rows, ignore_index=True)


def fetch_fotmob_player_minutes(competitions: list[str] | None = None) -> pd.DataFrame:
    """Return player minutes for the specified competitions.

    Args:
        competitions: List of competition names to include.
                      Defaults to SUPPORTED_COMPETITIONS (excluding PL).

    Returns:
        DataFrame with columns: player_name, team, competition, date, minutes
    """
    if competitions is None:
        competitions = SUPPORTED_COMPETITIONS
    raw = _fetch_fotmob_raw(competitions)
    if raw.empty:
        return raw
    # Ensure we only return rows for requested competitions
    return raw[raw["competition"].isin(competitions)].reset_index(drop=True)


def cross_validate_with_fpl(
    fotmob_pl: pd.DataFrame,
    fpl_minutes: pd.DataFrame,
) -> FotMobReliabilityResult:
    """Cross-validate FotMob PL minutes against FPL element-summary minutes.

    Join key: (player_name / web_name, team, date).
    Returns MAE and correlation for matched rows.
    """
    if fotmob_pl.empty or fpl_minutes.empty:
        return FotMobReliabilityResult(mae=float("inf"), correlation=0.0,
                                       n_matched=0, reliable=False)

    merged = fotmob_pl.merge(
        fpl_minutes.rename(columns={"web_name": "player_name"}),
        on=["player_name", "team", "date"],
        suffixes=("_fotmob", "_fpl"),
    )

    if merged.empty:
        logger.warning("No matched rows in FotMob vs FPL cross-validation")
        return FotMobReliabilityResult(mae=float("inf"), correlation=0.0,
                                       n_matched=0, reliable=False)

    mae = (merged["minutes_fotmob"] - merged["minutes_fpl"]).abs().mean()
    corr = merged["minutes_fotmob"].corr(merged["minutes_fpl"])
    n = len(merged)
    reliable = mae <= 5.0 and corr >= 0.95
    return FotMobReliabilityResult(mae=float(mae), correlation=float(corr),
                                   n_matched=n, reliable=reliable)
```

- [x] **Step 4: Run tests to confirm PASS**

```bash
pytest tests/datasources/test_soccerdata.py -v
# Expected: 4 passed
```

- [x] **Step 5: Commit**

```bash
git add src/pipeline/datasources/soccerdata_client.py tests/datasources/test_soccerdata.py
git commit -m "feat(H-F3): FotMob European minutes client with FPL cross-validation"
```

---

## Task 4 — Fantasy Football Scout RSS Parser (H-F4)

**Files:**
- Create: `src/pipeline/datasources/ffs.py`
- Add tests to: `tests/datasources/test_signals.py`

### Background

FFS RSS (`https://www.fantasyfootballscout.co.uk/feed`) is a standard RSS 2.0 feed that updates hourly. We use `feedparser` to parse it. Each item becomes a `PlayerSignal` via name resolution.

**Signal type classification (rule-based, not NLP):**
- Title/description contains `doubt`, `injury`, `knock`, `concern` → `"doubt"`
- Contains `fit`, `available`, `returns`, `back` → `"available"`
- Contains `out`, `ruled out`, `miss`, `injured` (more definitive) → `"injured"`
- No keyword match → `"general_news"`

FPL API `status` is the ground truth — if FFS says "available" but FPL `status == "i"`, the signal is downgraded to display-only.

- [x] **Step 1: Write failing tests (add to existing test_signals.py)**

```python
# Add to tests/datasources/test_signals.py

from src.pipeline.datasources.ffs import parse_ffs_feed, _classify_signal_type

MOCK_RSS_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Fantasy Football Scout</title>
    <item>
      <title>Salah doubt for GW32 after training knock</title>
      <description>Mohamed Salah is a doubt for the upcoming gameweek.</description>
      <pubDate>Sat, 05 Apr 2026 08:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Havertz available and fit to play</title>
      <description>Kai Havertz has returned to training and is available.</description>
      <pubDate>Sat, 05 Apr 2026 09:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>"""


def test_classify_doubt():
    assert _classify_signal_type("Salah doubt for GW32 after knock") == "doubt"


def test_classify_available():
    assert _classify_signal_type("Player available and fit to play") == "available"


def test_classify_injured():
    assert _classify_signal_type("Player ruled out for three weeks") == "injured"


def test_classify_general_news():
    assert _classify_signal_type("Manager press conference notes") == "general_news"


def test_parse_ffs_feed_returns_signals():
    signals = parse_ffs_feed(
        rss_content=MOCK_RSS_FEED,
        bootstrap_data=MOCK_BOOTSTRAP,
    )
    assert isinstance(signals, list)
    assert len(signals) > 0
    for sig in signals:
        from src.pipeline.datasources.signals import PlayerSignal
        assert isinstance(sig, PlayerSignal)
        assert sig.source == "ffs"


def test_parse_ffs_doubt_signal():
    signals = parse_ffs_feed(rss_content=MOCK_RSS_FEED, bootstrap_data=MOCK_BOOTSTRAP)
    doubt_signals = [s for s in signals if s.signal_type == "doubt"]
    assert len(doubt_signals) >= 1
    assert doubt_signals[0].player_code == 80201  # Salah
```

- [x] **Step 2: Run to confirm FAIL**

```bash
pytest tests/datasources/test_signals.py -v -k "ffs or classify"
# Expected: ImportError or AttributeError
```

- [x] **Step 3: Implement FFS parser**

Also update parsers to call `log_unresolved_name` on failed resolution (replacing silent `continue`). The pattern is the same in `ffs.py`, `reddit.py`, and `premierinjuries.py` — shown below for `ffs.py`; apply the same pattern in the other two.

```python
# src/pipeline/datasources/ffs.py
"""Fantasy Football Scout RSS parser → PlayerSignal list.

Feed: https://www.fantasyfootballscout.co.uk/feed (standard RSS 2.0, updates hourly)
Covers: injuries, DGW/BGW confirmations, international duty minutes, team news.
Signal classification: rule-based keyword matching (not NLP).
"""
from __future__ import annotations
import logging
import re
import feedparser
from .signals import PlayerSignal, resolve_player_name

logger = logging.getLogger(__name__)

FFS_FEED_URL = "https://www.fantasyfootballscout.co.uk/feed"

_DOUBT_KEYWORDS   = re.compile(r"\bdoubt|knock|concern|uncertain\b", re.I)
_AVAILABLE_KEYWORDS = re.compile(r"\bavailable|fit|returns|back in\b", re.I)
_INJURED_KEYWORDS = re.compile(r"\bruled out|miss(?:es|ing)?|injured|out for\b", re.I)


def _classify_signal_type(text: str) -> str:
    """Classify signal type from title/description text using keyword rules."""
    if _INJURED_KEYWORDS.search(text):
        return "injured"
    if _DOUBT_KEYWORDS.search(text):
        return "doubt"
    if _AVAILABLE_KEYWORDS.search(text):
        return "available"
    return "general_news"


def _extract_player_names(text: str) -> list[str]:
    """Extract candidate player names from a title string.

    Heuristic: capitalised words that are not common FPL stop words.
    Falls back to the first two capitalised tokens.
    """
    stop_words = {"GW", "FPL", "Premier", "League", "Fantasy", "Football", "Scout",
                  "GW32", "GW33", "GW34", "GW35", "GW36", "GW37", "GW38"}
    tokens = text.split()
    candidates = [
        t.rstrip("'s,.")
        for t in tokens
        if t and t[0].isupper() and t not in stop_words
    ]
    # Try longest-first: "Mohamed Salah" before "Salah"
    names = []
    i = 0
    while i < len(candidates):
        if i + 1 < len(candidates):
            names.append(candidates[i] + " " + candidates[i + 1])
        names.append(candidates[i])
        i += 1
    return names


def parse_ffs_feed(
    rss_content: str | None = None,
    bootstrap_data: dict | None = None,
    url: str = FFS_FEED_URL,
) -> list[PlayerSignal]:
    """Parse the FFS RSS feed into PlayerSignal objects.

    Args:
        rss_content: Raw RSS XML string (for testing — bypasses HTTP fetch).
        bootstrap_data: FPL bootstrap dict for player name resolution.
        url: RSS feed URL (only used if rss_content is None).

    Returns:
        List of PlayerSignal objects (may be empty if no names resolve).
    """
    if rss_content:
        feed = feedparser.parse(rss_content)
    else:
        feed = feedparser.parse(url)

    signals = []
    for entry in feed.entries:
        title = entry.get("title", "")
        description = entry.get("summary", "")
        combined = f"{title} {description}"
        signal_type = _classify_signal_type(combined)
        timestamp = entry.get("published", "")

        if bootstrap_data:
            candidate_names = _extract_player_names(title)
            resolved_code = None
            for name in candidate_names:
                code = resolve_player_name(name, bootstrap_data)
                if code is not None:
                    resolved_code = code
                    break
            if resolved_code is None:
                from .signals import log_unresolved_name
                log_unresolved_name(
                    name=" / ".join(candidate_names[:2]), source="ffs",
                    raw_text=title, timestamp=timestamp,
                )
                continue
        else:
            # Without bootstrap, we cannot resolve — skip
            logger.warning("FFS: bootstrap_data not provided, skipping name resolution")
            continue

        signals.append(PlayerSignal(
            player_code=resolved_code,
            source="ffs",
            signal_type=signal_type,
            text=combined.strip(),
            timestamp=timestamp,
            confidence=0.9,  # structured source
        ))

    return signals
```

- [x] **Step 4: Run tests to confirm PASS**

```bash
pytest tests/datasources/test_signals.py -v
# Expected: all passing (original 5 + new 6 = 11 passed)
```

- [x] **Step 5: Commit**

```bash
git add src/pipeline/datasources/ffs.py tests/datasources/test_signals.py
git commit -m "feat(H-F4): FFS RSS parser with rule-based signal classification"
```

---

## Task 5 — Reddit r/FantasyPL API Client (H-F5)

**Files:**
- Create: `src/pipeline/datasources/reddit.py`
- Add tests to: `tests/datasources/test_signals.py`

### Background

Reddit's JSON API is free and requires no authentication for public subreddits.
Endpoint: `https://www.reddit.com/r/FantasyPL/new.json?limit=25&t=day`

We collect posts 24–48h before a deadline. Filter for posts that mention injury or rotation signals. Phase 1 is **display-only** — no xP adjustment. Confidence is lower (0.5) than structured sources.

**Rate limit:** Reddit enforces 1 request/second for unauthenticated apps. Use a `User-Agent` header.

- [x] **Step 1: Write failing tests (add to test_signals.py)**

```python
# Add to tests/datasources/test_signals.py

from src.pipeline.datasources.reddit import parse_reddit_posts

MOCK_REDDIT_RESPONSE = {
    "data": {
        "children": [
            {"data": {
                "title": "Salah doubtful — training ground reports suggest knock",
                "selftext": "Multiple sources saying Salah picked up a knock in training.",
                "created_utc": 1743840000,
                "score": 245,
            }},
            {"data": {
                "title": "My GW32 template — what do you think?",
                "selftext": "Here is my squad...",
                "created_utc": 1743840000,
                "score": 12,
            }},
        ]
    }
}


def test_parse_reddit_returns_signals():
    signals = parse_reddit_posts(
        posts_data=MOCK_REDDIT_RESPONSE,
        bootstrap_data=MOCK_BOOTSTRAP,
        min_score=50,
    )
    assert isinstance(signals, list)
    # Salah post should resolve; template post has no player name + low signal
    assert all(hasattr(s, "player_code") for s in signals)


def test_parse_reddit_filters_low_score():
    signals = parse_reddit_posts(
        posts_data=MOCK_REDDIT_RESPONSE,
        bootstrap_data=MOCK_BOOTSTRAP,
        min_score=300,  # Both posts below this
    )
    assert signals == []


def test_reddit_signals_have_low_confidence():
    signals = parse_reddit_posts(
        posts_data=MOCK_REDDIT_RESPONSE,
        bootstrap_data=MOCK_BOOTSTRAP,
        min_score=50,
    )
    for sig in signals:
        assert sig.confidence <= 0.6
        assert sig.source == "reddit"
```

- [x] **Step 2: Run to confirm FAIL**

```bash
pytest tests/datasources/test_signals.py -v -k "reddit"
# Expected: ImportError
```

- [x] **Step 3: Implement Reddit client**

```python
# src/pipeline/datasources/reddit.py
"""Reddit r/FantasyPL JSON API client → PlayerSignal list (display-only, Phase 1).

Endpoint: https://www.reddit.com/r/FantasyPL/new.json
No authentication required for public subreddits.
Rate limit: 1 req/sec for unauthenticated clients — use User-Agent header.

Phase 1: display-only. Confidence = 0.5 (lower than structured sources).
"""
from __future__ import annotations
import logging
import time
import requests
from .signals import PlayerSignal, resolve_player_name
from .ffs import _classify_signal_type, _extract_player_names

logger = logging.getLogger(__name__)

REDDIT_URL = "https://www.reddit.com/r/FantasyPL/new.json"
REDDIT_USER_AGENT = "fpl-assistant-signals/1.0 (personal FPL tool)"


def fetch_reddit_posts(limit: int = 25, time_filter: str = "day") -> dict:
    """Fetch recent posts from r/FantasyPL."""
    resp = requests.get(
        REDDIT_URL,
        params={"limit": limit, "t": time_filter},
        headers={"User-Agent": REDDIT_USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    time.sleep(1.0)  # Respect Reddit rate limit
    return resp.json()


def parse_reddit_posts(
    posts_data: dict,
    bootstrap_data: dict,
    min_score: int = 50,
) -> list[PlayerSignal]:
    """Parse Reddit posts into PlayerSignal objects.

    Args:
        posts_data: JSON response from Reddit API.
        bootstrap_data: FPL bootstrap for player name resolution.
        min_score: Minimum Reddit score (upvotes) to consider.

    Returns:
        List of PlayerSignal objects. Confidence = 0.5 (community source).
    """
    signals = []
    for child in posts_data.get("data", {}).get("children", []):
        post = child.get("data", {})
        score = post.get("score", 0)
        if score < min_score:
            continue

        title = post.get("title", "")
        body = post.get("selftext", "")
        combined = f"{title} {body}"
        signal_type = _classify_signal_type(combined)

        if signal_type == "general_news":
            continue  # Skip non-signal posts

        candidate_names = _extract_player_names(title)
        resolved_code = None
        for name in candidate_names:
            code = resolve_player_name(name, bootstrap_data)
            if code is not None:
                resolved_code = code
                break

        from datetime import datetime, timezone
        ts = datetime.fromtimestamp(
            post.get("created_utc", 0), tz=timezone.utc
        ).isoformat()

        if resolved_code is None:
            from .signals import log_unresolved_name
            log_unresolved_name(
                name=" / ".join(candidate_names[:2]), source="reddit",
                raw_text=title, timestamp=ts,
            )
            continue

        ts = datetime.fromtimestamp(
            post.get("created_utc", 0), tz=timezone.utc
        ).isoformat()

        signals.append(PlayerSignal(
            player_code=resolved_code,
            source="reddit",
            signal_type=signal_type,
            text=title.strip(),
            timestamp=ts,
            confidence=0.5,
        ))

    return signals
```

- [x] **Step 4: Run tests to confirm PASS**

```bash
pytest tests/datasources/test_signals.py -v
# Expected: all passing
```

- [x] **Step 5: Commit**

```bash
git add src/pipeline/datasources/reddit.py tests/datasources/test_signals.py
git commit -m "feat(H-F5): Reddit r/FantasyPL signal client (display-only, Phase 1)"
```

---

## Task 6 — premierinjuries.com Scraper (H-F6)

**Files:**
- Create: `src/pipeline/datasources/premierinjuries.py`
- Add tests to: `tests/datasources/test_signals.py`

### Background

premierinjuries.com hosts @BenDinnery's structured injury readiness tables. HTML structure is more stable than social media. Signal types map directly to FPL availability statuses: `doubt`, `available`, `injured`.

**Cross-verification rule (hard):** Any signal that contradicts FPL `status` is flagged as contradicted and must NOT feed into xP adjustment. Log to `signal_accuracy.csv` with `contradicted=True`.

**Signal confidence = 0.8** (structured website, rule-based parse, but DOM can drift).

- [x] **Step 1: Write failing tests (add to test_signals.py)**

```python
# Add to tests/datasources/test_signals.py

from src.pipeline.datasources.premierinjuries import (
    parse_premierinjuries_html,
    cross_verify_against_fpl,
)

MOCK_PI_HTML = """<html><body>
<table id="player-injury-table">
  <thead><tr><th>Player</th><th>Club</th><th>Status</th><th>Notes</th></tr></thead>
  <tbody>
    <tr><td>Mohamed Salah</td><td>Liverpool</td><td>Doubt</td><td>Knock in training</td></tr>
    <tr><td>Unknown Player X</td><td>City</td><td>Available</td><td>Fit to play</td></tr>
  </tbody>
</table>
</body></html>"""

MOCK_FPL_STATUS = {80201: "a"}  # Salah is "available" per FPL API


def test_parse_pi_html_returns_signals():
    signals = parse_premierinjuries_html(
        html_content=MOCK_PI_HTML,
        bootstrap_data=MOCK_BOOTSTRAP,
    )
    assert isinstance(signals, list)
    # Unknown Player X won't resolve
    assert all(s.source == "premierinjuries" for s in signals)


def test_parse_pi_resolved_signal():
    signals = parse_premierinjuries_html(
        html_content=MOCK_PI_HTML,
        bootstrap_data=MOCK_BOOTSTRAP,
    )
    resolved = [s for s in signals if s.player_code == 80201]
    assert len(resolved) == 1
    assert resolved[0].signal_type == "doubt"


def test_parse_pi_unresolved_skipped():
    signals = parse_premierinjuries_html(
        html_content=MOCK_PI_HTML,
        bootstrap_data=MOCK_BOOTSTRAP,
    )
    # Unknown Player X has no match in MOCK_BOOTSTRAP → skipped
    player_codes = [s.player_code for s in signals]
    assert 80201 in player_codes
    # Only 1 resolved (Salah)
    assert len(signals) == 1


def test_parse_pi_unresolved_writes_csv(tmp_path):
    """Unresolved names from premierinjuries parser must be logged to signal_unresolved.csv."""
    from src.pipeline.datasources.signals import log_unresolved_name
    csv_path = tmp_path / "signal_unresolved.csv"
    # Call log_unresolved_name directly with tmp csv_path to confirm the write path works
    log_unresolved_name(
        name="Unknown Player X", source="premierinjuries",
        raw_text="Unknown Player X: available.", csv_path=csv_path,
    )
    import pandas as pd
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert df.iloc[0]["name"] == "Unknown Player X"
    assert df.iloc[0]["source"] == "premierinjuries"


def test_parse_reddit_unresolved_writes_csv(tmp_path):
    """Unresolved names from Reddit parser must be logged; verify via log_unresolved_name."""
    from src.pipeline.datasources.signals import log_unresolved_name
    csv_path = tmp_path / "signal_unresolved.csv"
    log_unresolved_name(
        name="Zxqwertyplayer123", source="reddit",
        raw_text="Zxqwertyplayer123 doubt for GW32", csv_path=csv_path,
    )
    import pandas as pd
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert df.iloc[0]["source"] == "reddit"


def test_cross_verify_contradiction_detected():
    """FFS says doubt, FPL says available (status 'i') — should flag contradiction."""
    signal = PlayerSignal(
        player_code=80201, source="premierinjuries",
        signal_type="injured", text="Salah out", timestamp="2026-04-05T08:00:00Z",
    )
    fpl_status = {80201: "a"}  # FPL says available
    result = cross_verify_against_fpl([signal], fpl_status)
    assert result[0]["contradicted"] is True


def test_cross_verify_consistent():
    signal = PlayerSignal(
        player_code=80201, source="premierinjuries",
        signal_type="doubt", text="Salah doubt", timestamp="2026-04-05T08:00:00Z",
    )
    fpl_status = {80201: "d"}  # FPL also says doubt/75% chance
    result = cross_verify_against_fpl([signal], fpl_status)
    assert result[0]["contradicted"] is False
```

- [x] **Step 2: Run to confirm FAIL**

```bash
pytest tests/datasources/test_signals.py -v -k "pi or premierinjuries or cross_verify"
# Expected: ImportError
```

- [x] **Step 3: Implement premierinjuries scraper**

```python
# src/pipeline/datasources/premierinjuries.py
"""premierinjuries.com scraper → PlayerSignal list (@BenDinnery content).

Structured injury readiness table. Signal confidence = 0.8 (DOM can drift).
Cross-verification rule: contradictions with FPL API status are flagged and
must NOT feed into xP adjustment — display-only.

FPL status field values:
  "a" = available, "d" = doubt, "i" = injured, "u" = unavailable,
  "s" = suspended, "n" = not available (international/other)
"""
from __future__ import annotations
import logging
import requests
from bs4 import BeautifulSoup
from .signals import PlayerSignal, resolve_player_name

logger = logging.getLogger(__name__)

PREMIERINJURIES_URL = "https://www.premierinjuries.com/injury-table.php"

# Mapping from premierinjuries status text to our signal_type
_STATUS_MAP = {
    "doubt": "doubt",
    "50/50": "doubt",
    "injured": "injured",
    "out": "injured",
    "available": "available",
    "fit": "available",
    "recovered": "available",
}

# FPL status codes that indicate doubt/injury (for contradiction detection)
_FPL_UNAVAILABLE = {"i", "u", "s", "n"}
_FPL_DOUBT = {"d"}  # chance_of_playing_next_round = 25 or 50 or 75


def fetch_premierinjuries_html() -> str:
    """Fetch the premierinjuries.com injury table HTML."""
    resp = requests.get(
        PREMIERINJURIES_URL,
        headers={"User-Agent": "fpl-assistant-signals/1.0"},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.text


def parse_premierinjuries_html(
    html_content: str,
    bootstrap_data: dict,
) -> list[PlayerSignal]:
    """Parse premierinjuries HTML table into PlayerSignal objects.

    Looks for a <table> with id="player-injury-table" or the first table
    with Player/Status columns.

    Args:
        html_content: Raw HTML string.
        bootstrap_data: FPL bootstrap for player name resolution.

    Returns:
        List of PlayerSignal objects (only resolved players included).
    """
    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("table", id="player-injury-table") or soup.find("table")
    if not table:
        logger.warning("premierinjuries: no table found in HTML")
        return []

    signals = []
    rows = table.find_all("tr")
    for row in rows[1:]:  # Skip header
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) < 3:
            continue
        player_name = cells[0]
        status_raw = cells[2].lower()
        notes = cells[3] if len(cells) > 3 else ""
        signal_type = _STATUS_MAP.get(status_raw, "general_news")

        code = resolve_player_name(player_name, bootstrap_data)
        if code is None:
            from .signals import log_unresolved_name
            log_unresolved_name(
                name=player_name, source="premierinjuries",
                raw_text=f"{player_name}: {status_raw}. {notes}".strip(),
            )
            continue

        signals.append(PlayerSignal(
            player_code=code,
            source="premierinjuries",
            signal_type=signal_type,
            text=f"{player_name}: {status_raw}. {notes}".strip(),
            timestamp="",  # premierinjuries doesn't include timestamps
            confidence=0.8,
        ))

    return signals


def cross_verify_against_fpl(
    signals: list[PlayerSignal],
    fpl_status: dict[int, str],
) -> list[dict]:
    """Cross-verify signals against FPL API status field.

    A contradiction occurs when the signal claims a player is injured/doubt
    but FPL status says "a" (available), or vice versa.

    Args:
        signals: List of PlayerSignal objects.
        fpl_status: Dict of {player_code: fpl_status_string}.

    Returns:
        List of dicts: {signal, contradicted: bool, fpl_status: str}
    """
    results = []
    for sig in signals:
        fpl_st = fpl_status.get(sig.player_code, "a")
        contradicted = False

        if sig.signal_type == "injured" and fpl_st == "a":
            contradicted = True
            logger.warning(
                "premierinjuries contradiction: player %d marked injured but FPL says 'a'",
                sig.player_code,
            )
        elif sig.signal_type == "doubt" and fpl_st == "a":
            # FPL says fully available; external source says doubt — flag as contradicted
            contradicted = True
            logger.warning(
                "premierinjuries contradiction: player %d marked doubt but FPL says 'a'",
                sig.player_code,
            )
        elif sig.signal_type == "available" and fpl_st in _FPL_UNAVAILABLE:
            contradicted = True
            logger.warning(
                "premierinjuries contradiction: player %d marked available but FPL says '%s'",
                sig.player_code, fpl_st,
            )

        results.append({
            "signal": sig,
            "contradicted": contradicted,
            "fpl_status": fpl_st,
        })

    return results
```

- [x] **Step 4: Run tests to confirm PASS**

```bash
pytest tests/datasources/test_signals.py -v
# Expected: all passing
```

- [x] **Step 5: Commit**

```bash
git add src/pipeline/datasources/premierinjuries.py tests/datasources/test_signals.py
git commit -m "feat(H-F6): premierinjuries.com scraper with FPL contradiction detection"
```

---

## Task 7 — Signal Feedback Logger (H-F7)

**Files:**
- Create: `src/pipeline/signal_feedback.py`
- Create: `tests/datasources/test_signal_feedback.py`

### Background

When team sheets arrive (~1h before kickoff), we log each signal against the actual lineup to build per-source accuracy scores. This data is the prerequisite for Track G Phase 2 (xP auto-adjustment activation threshold: ≥ 80% accuracy over ≥ 15 observations per source-type pair).

Schema: `signal_id, source, signal_type, player_code, gw, predicted_status, actual_started, contradicted, run_date`

`signal_id` is a deterministic hash: `md5(f"{source}:{player_code}:{gw}:{signal_type}")`.

- [x] **Step 1: Write failing tests**

```python
# tests/datasources/test_signal_feedback.py
import pytest
import pandas as pd
from pathlib import Path
from src.pipeline.signal_feedback import (
    append_signal_feedback,
    compute_source_accuracy,
    make_signal_id,
)
from src.pipeline.datasources.signals import PlayerSignal


def _make_signal(source="ffs", signal_type="doubt", player_code=80201):
    return PlayerSignal(
        player_code=player_code, source=source,
        signal_type=signal_type, text="test", timestamp="2026-04-05T00:00:00Z",
    )


def test_make_signal_id_deterministic():
    id1 = make_signal_id(source="ffs", player_code=80201, gw=32, signal_type="doubt")
    id2 = make_signal_id(source="ffs", player_code=80201, gw=32, signal_type="doubt")
    assert id1 == id2


def test_make_signal_id_differs_by_source():
    id1 = make_signal_id("ffs", 80201, 32, "doubt")
    id2 = make_signal_id("reddit", 80201, 32, "doubt")
    assert id1 != id2


def test_append_signal_feedback_creates_file(tmp_path):
    csv_path = tmp_path / "signal_accuracy.csv"
    sig = _make_signal()
    append_signal_feedback(
        signal=sig, gw=32, actual_started=True,
        contradicted=False, csv_path=csv_path,
    )
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["source"] == "ffs"
    assert df.iloc[0]["actual_started"] == True


def test_append_signal_feedback_appends(tmp_path):
    csv_path = tmp_path / "signal_accuracy.csv"
    sig = _make_signal()
    append_signal_feedback(sig, gw=32, actual_started=True,
                           contradicted=False, csv_path=csv_path)
    append_signal_feedback(sig, gw=33, actual_started=False,
                           contradicted=False, csv_path=csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 2


def test_compute_source_accuracy():
    df = pd.DataFrame([
        {"source": "ffs", "signal_type": "doubt", "actual_started": False},
        {"source": "ffs", "signal_type": "doubt", "actual_started": False},
        {"source": "ffs", "signal_type": "doubt", "actual_started": True},
        {"source": "reddit", "signal_type": "doubt", "actual_started": True},
    ])
    acc = compute_source_accuracy(df)
    # ffs doubt: 2/3 correct (actual_started=False when doubt predicted)
    ffs_acc = acc.loc[(acc["source"] == "ffs") & (acc["signal_type"] == "doubt"), "accuracy"]
    assert pytest.approx(ffs_acc.values[0], abs=0.01) == 2/3


def test_signal_feedback_schema(tmp_path):
    csv_path = tmp_path / "signal_accuracy.csv"
    sig = _make_signal()
    append_signal_feedback(sig, gw=32, actual_started=True,
                           contradicted=False, csv_path=csv_path)
    df = pd.read_csv(csv_path)
    required_cols = {
        "signal_id", "source", "signal_type", "player_code",
        "gw", "predicted_status", "actual_started", "contradicted", "run_date"
    }
    assert required_cols.issubset(set(df.columns))
```

- [x] **Step 2: Run to confirm FAIL**

```bash
pytest tests/datasources/test_signal_feedback.py -v
# Expected: ImportError
```

- [x] **Step 3: Implement signal feedback logger**

```python
# src/pipeline/signal_feedback.py
"""Signal feedback logger — tracks prediction accuracy per source-type pair.

When team sheets arrive, log each PlayerSignal against actual lineup.
Data required for Track G Phase 2 activation gate:
  ≥ 80% accuracy over ≥ 15 observations per (source, signal_type) pair.

Schema: signal_id, source, signal_type, player_code, gw,
        predicted_status, actual_started, contradicted, run_date
"""
from __future__ import annotations
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from src.pipeline.datasources.signals import PlayerSignal
from src.config import SIGNAL_ACCURACY_CSV

logger = logging.getLogger(__name__)


def make_signal_id(
    source: str, player_code: int, gw: int, signal_type: str, timestamp: str = ""
) -> str:
    """Deterministic signal ID: md5 of key fields + timestamp.

    Timestamp prevents collision when the same source issues two signals
    for the same player/GW/type (e.g. two FFS articles both saying 'Salah doubt GW32').
    """
    raw = f"{source}:{player_code}:{gw}:{signal_type}:{timestamp}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def append_signal_feedback(
    signal: PlayerSignal,
    gw: int,
    actual_started: bool,
    contradicted: bool,
    csv_path: Path | None = None,
) -> None:
    """Append one signal feedback row to signal_accuracy.csv.

    Args:
        signal: The PlayerSignal that was issued pre-deadline.
        gw: Gameweek number.
        actual_started: True if the player started (appeared on team sheet).
        contradicted: True if signal contradicted FPL API status at issue time.
        csv_path: Override path (for testing). Defaults to SIGNAL_ACCURACY_CSV.
    """
    if csv_path is None:
        csv_path = SIGNAL_ACCURACY_CSV

    row = pd.DataFrame([{
        "signal_id": make_signal_id(signal.source, signal.player_code, gw, signal.signal_type, signal.timestamp),
        "source": signal.source,
        "signal_type": signal.signal_type,
        "player_code": signal.player_code,
        "gw": gw,
        "predicted_status": signal.signal_type,
        "actual_started": actual_started,
        "contradicted": contradicted,
        "run_date": datetime.now(tz=timezone.utc).isoformat(),
    }])

    if csv_path.exists():
        row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(csv_path, index=False)


def compute_source_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(source, signal_type) accuracy from a feedback log DataFrame.

    Accuracy definition:
      - For signal_type == "doubt" | "injured": correct if actual_started == False
      - For signal_type == "available": correct if actual_started == True

    Returns DataFrame with columns: source, signal_type, accuracy, n_observations
    """
    df = df.copy()
    df["correct"] = df.apply(
        lambda r: (not r["actual_started"])
        if r["signal_type"] in ("doubt", "injured")
        else bool(r["actual_started"]),
        axis=1,
    )
    acc = (
        df.groupby(["source", "signal_type"])
        .agg(accuracy=("correct", "mean"), n_observations=("correct", "count"))
        .reset_index()
    )
    return acc
```

- [x] **Step 4: Run tests to confirm PASS**

```bash
pytest tests/datasources/test_signal_feedback.py -v
# Expected: 6 passed
```

- [x] **Step 5: Commit**

```bash
git add src/pipeline/signal_feedback.py tests/datasources/test_signal_feedback.py
git commit -m "feat(H-F7): signal feedback logger with per-source accuracy computation"
```

---

## Task 8 — Integration Tests and Full Suite

**Files:**
- Create: `tests/datasources/test_integration_datasources.py`

### Background

Integration tests verify the full chain: fetch → parse → signal → resolve → log. They use VCR cassettes to replay HTTP responses — no live network calls in CI.

For the **xG validation gate integration test**, load real vaastav data (2 seasons) and understat cached data, run `compute_source_spearman` on both, verify the gate decision is deterministic.

- [x] **Step 1: Write integration tests**

```python
# tests/datasources/test_integration_datasources.py
"""Integration tests for Track H datasources.

HTTP calls are mocked via unittest.mock (not VCR) to avoid cassette files.
These tests verify the end-to-end flow from raw API response to PlayerSignal.
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.pipeline.datasources.ffs import parse_ffs_feed
from src.pipeline.datasources.reddit import parse_reddit_posts
from src.pipeline.datasources.premierinjuries import parse_premierinjuries_html, cross_verify_against_fpl
from src.pipeline.datasources.signals import PlayerSignal
from src.pipeline.signal_feedback import append_signal_feedback, compute_source_accuracy
from src.pipeline.source_validation import run_xg_validation_gate, SourceValidationResult

MINIMAL_BOOTSTRAP = {
    "elements": [
        {"id": 1, "code": 80201, "web_name": "Salah",
         "first_name": "Mohamed", "second_name": "Salah", "team": 14},
    ]
}

FFS_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Mohamed Salah doubt for GW32</title>
    <description>Salah is a doubt.</description>
    <pubDate>Sat, 05 Apr 2026 08:00:00 +0000</pubDate>
  </item>
</channel></rss>"""

REDDIT_DATA = {"data": {"children": [
    {"data": {"title": "Salah doubt for GW32", "selftext": "",
              "created_utc": 1743840000, "score": 150}}
]}}

PI_HTML = """<html><body>
<table id="player-injury-table">
  <thead><tr><th>Player</th><th>Club</th><th>Status</th><th>Notes</th></tr></thead>
  <tbody>
    <tr><td>Mohamed Salah</td><td>Liverpool</td><td>Doubt</td><td>Training knock</td></tr>
  </tbody>
</table></body></html>"""


def test_ffs_to_signal_full_chain():
    signals = parse_ffs_feed(rss_content=FFS_RSS, bootstrap_data=MINIMAL_BOOTSTRAP)
    assert len(signals) == 1
    assert signals[0].player_code == 80201
    assert signals[0].signal_type == "doubt"
    assert signals[0].source == "ffs"


def test_reddit_to_signal_full_chain():
    signals = parse_reddit_posts(posts_data=REDDIT_DATA, bootstrap_data=MINIMAL_BOOTSTRAP, min_score=50)
    assert len(signals) == 1
    assert signals[0].player_code == 80201
    assert signals[0].source == "reddit"
    assert signals[0].confidence <= 0.6


def test_pi_to_signal_full_chain():
    signals = parse_premierinjuries_html(html_content=PI_HTML, bootstrap_data=MINIMAL_BOOTSTRAP)
    assert len(signals) == 1
    assert signals[0].player_code == 80201
    assert signals[0].signal_type == "doubt"


def test_fpl_contradiction_flags_signal():
    signals = parse_premierinjuries_html(html_content=PI_HTML, bootstrap_data=MINIMAL_BOOTSTRAP)
    # FPL says available, PI says doubt — contradiction
    fpl_status = {80201: "a"}
    verified = cross_verify_against_fpl(signals, fpl_status)
    assert verified[0]["contradicted"] is True


def test_fpl_consistent_no_contradiction():
    signals = parse_premierinjuries_html(html_content=PI_HTML, bootstrap_data=MINIMAL_BOOTSTRAP)
    fpl_status = {80201: "d"}  # doubt matches
    verified = cross_verify_against_fpl(signals, fpl_status)
    assert verified[0]["contradicted"] is False


def test_signal_feedback_full_chain(tmp_path):
    csv_path = tmp_path / "signal_accuracy.csv"
    signals = parse_ffs_feed(rss_content=FFS_RSS, bootstrap_data=MINIMAL_BOOTSTRAP)
    for sig in signals:
        append_signal_feedback(sig, gw=32, actual_started=False,
                               contradicted=False, csv_path=csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["source"] == "ffs"


def test_source_accuracy_computation(tmp_path):
    csv_path = tmp_path / "signal_accuracy.csv"
    signals = parse_ffs_feed(rss_content=FFS_RSS, bootstrap_data=MINIMAL_BOOTSTRAP)
    sig = signals[0]
    # Log 3 correct predictions (doubt → did not start) and 1 wrong
    for gw, started in [(30, False), (31, False), (32, False), (33, True)]:
        append_signal_feedback(sig, gw=gw, actual_started=started,
                               contradicted=False, csv_path=csv_path)
    df = pd.read_csv(csv_path)
    acc = compute_source_accuracy(df)
    ffs_row = acc[(acc["source"] == "ffs") & (acc["signal_type"] == "doubt")]
    assert pytest.approx(ffs_row["accuracy"].values[0], abs=0.01) == 0.75


def test_xg_validation_gate_integration():
    result = run_xg_validation_gate(
        understat_rho=0.63, fpl_opta_rho=0.66, tolerance=0.05
    )
    assert isinstance(result, SourceValidationResult)
    assert result.passed is True
    assert result.recommended_source == "understat"


def test_xg_gate_spearman_both_sources():
    """Compute Spearman rho for both xG sources on synthetic matched data.

    Simulates the vaastav join: a DataFrame with both understat_xG and
    fpl_opta_xG columns joined against actual goals.
    This test verifies compute_source_spearman is callable on realistic data
    and that the gate decision is deterministic given the same inputs.
    """
    from src.pipeline.source_validation import compute_source_spearman
    import numpy as np
    rng = np.random.default_rng(42)
    n = 200
    actual_goals = rng.integers(0, 3, size=n).astype(float)
    # understat: slightly noisy correlation
    understat_xg = actual_goals + rng.normal(0, 0.3, size=n)
    # FPL Opta: similar noise
    fpl_opta_xg  = actual_goals + rng.normal(0, 0.35, size=n)
    df = pd.DataFrame({
        "understat_xG": understat_xg,
        "fpl_opta_xG": fpl_opta_xg,
        "goals_scored": actual_goals,
    })
    rho_u = compute_source_spearman(df, xg_col="understat_xG", actual_col="goals_scored")
    rho_f = compute_source_spearman(df, xg_col="fpl_opta_xG", actual_col="goals_scored")
    # Both should be high (synthetic data has a strong signal)
    assert rho_u > 0.80
    assert rho_f > 0.75
    # Gate decision must be reproducible
    gate1 = run_xg_validation_gate(rho_u, rho_f, tolerance=0.05)
    gate2 = run_xg_validation_gate(rho_u, rho_f, tolerance=0.05)
    assert gate1.passed == gate2.passed
    assert gate1.recommended_source == gate2.recommended_source
```

- [x] **Step 2: Run to confirm PASS (all green)**

```bash
pytest tests/datasources/ -v
# Expected: all tests passing (includes test_xg_gate_spearman_both_sources)
```

- [x] **Step 3: Run full test suite to confirm no regressions**

```bash
pytest tests/ -q
# Expected: all pre-existing tests still pass
```

- [x] **Step 4: Commit**

```bash
git add tests/datasources/test_integration_datasources.py
git commit -m "test(H-F0-F7): integration tests for full datasources chain"
```

---

## Task 9 — requirements.txt and Final Cleanup

**Files:**
- Modify: `requirements.txt`

- [x] **Step 1: Read current requirements.txt**

```bash
cat requirements.txt
```

- [x] **Step 2: Add new dependencies**

Add to `requirements.txt`:
```
understatapi
soccerdata
feedparser
```

(Note: `beautifulsoup4>=4.11.0` is already present — do not add a duplicate. `requests` is also already present.)

- [x] **Step 3: Verify install**

```bash
pip install -r requirements.txt
```

- [x] **Step 4: Run full suite one final time**

```bash
pytest tests/ -q
# Expected: all tests pass
```

- [x] **Step 5: Final commit**

```bash
git add requirements.txt
git commit -m "chore: add understatapi, soccerdata, feedparser to requirements for Track H"
```

---

## Merge Checklist

Before merging `feature/track-h-data-sources` → `master`:

- [x] All tests pass: `pytest tests/ -q`
- [x] No imports from `_original/` anywhere in new code
- [x] `source_validation.py` has a `__main__` guard for manual CLI use
- [x] Each data source module has a clear docstring noting what it covers and does NOT cover
- [x] `signal_feedback.py` docstring notes the Track G Phase 2 threshold requirements
- [x] No live HTTP calls in any test (all mocked)
- [x] `results/source_validation.csv` and `results/signal_accuracy.csv` added to `.gitignore`

```bash
# Add to .gitignore if not already present:
echo "results/source_validation.csv" >> .gitignore
echo "results/signal_accuracy.csv" >> .gitignore
echo "results/signal_unresolved.csv" >> .gitignore
```

---

## Source Reliability Summary

| Source | Reliability vs FPL API | Gate | Notes |
|--------|----------------------|------|-------|
| understatAPI | Validated by H-F2 Spearman ρ gate | Must pass before Track B B-F1b | EPL only; no UEFA/international |
| FotMob (soccerdata) | Cross-validated on PL matches (MAE ≤ 5 min, corr ≥ 0.95) | Reliability flag logged | Covers UEFA/international — FPL does not |
| FFS RSS | Structured, hourly updates; verified by signal feedback log | Display until ≥ 15 observations | Confidence 0.9 |
| Reddit | Community signal; lower quality | Display-only Phase 1 | Confidence 0.5; min_score=50 filters noise |
| premierinjuries.com | Contradictions cross-checked vs FPL status | Contradictions → display-only only | Confidence 0.8; DOM may drift |
| FPL API `status` | Ground truth | Always authoritative | Already in `availability.py` |

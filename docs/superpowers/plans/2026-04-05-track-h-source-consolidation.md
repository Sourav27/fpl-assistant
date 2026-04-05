# Track H — Source Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace two broken data source clients (understatapi async, FotMob), add ESPN as the non-PL competition source, build a unified availability feature module, and declare SOURCE_COLUMN_MAP as the authoritative column ownership registry.

**Architecture:** Five independent module changes in dependency order — `__init__.py` (registry) → `understat.py` (fix) → delete `soccerdata_client.py` + add `espn_client.py` (replacement) → `availability_features.py` (new) → `source_validation.py` (docstring). Each task is independently testable. No changes to the main pipeline run.py or predict.py — this is datasources layer only.

**Tech Stack:** Python, pandas, soccerdata (Understat synchronous), requests (ESPN direct API), difflib (fuzzy name match), pytest with HTTP mocking (unittest.mock.patch), existing `availability.py` HybridAvailabilityFilter wrapped internally.

**Working directory:** `.worktrees/track-h` (branch `feature/track-h-data-sources`)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/pipeline/datasources/__init__.py` | Modify | Add `SOURCE_COLUMN_MAP` export alongside `PlayerSignal` |
| `src/pipeline/datasources/understat.py` | Rewrite | Replace `understatapi` async → `soccerdata.Understat` sync; slim to `xg_chain`+`xg_buildup`; add date→GW join helper |
| `src/pipeline/datasources/soccerdata_client.py` | **Delete** | FotMob wrapper — broken, replaced by ESPN |
| `src/pipeline/datasources/espn_client.py` | **Create** | ESPN eventlog fetch, player ID resolver, caching, rate limiting, backfill |
| `src/pipeline/datasources/availability_features.py` | **Create** | Unified availability feature assembly; wraps `availability.py` |
| `src/pipeline/source_validation.py` | Docstring only | Clarify pipeline-root location; update gate description for xg_chain |
| `data/espn_player_id_map.csv` | **Create** | Seeded FPL code → ESPN ID lookup (~50-80 non-PL regulars) |
| `tests/datasources/test_soccerdata.py` | Rewrite | Replace FotMob tests with ESPN client tests (file renamed to `test_espn_client.py`) |
| `tests/datasources/test_understat.py` | Modify | Fix: patch soccerdata.Understat instead of understatapi async |
| `tests/datasources/test_availability_features.py` | **Create** | Unit tests for all availability assembly paths |

---

## Task 1: Add SOURCE_COLUMN_MAP to `__init__.py`

**Files:**
- Modify: `src/pipeline/datasources/__init__.py`
- Test: `tests/datasources/test_signals.py` (add one import check, no new file)

The `SOURCE_COLUMN_MAP` is the single declaration of what each source owns. It must be importable from the package root. No logic — pure data.

- [ ] **Step 1: Write failing import test**

Add to `tests/datasources/test_signals.py`:

```python
def test_source_column_map_importable():
    from src.pipeline.datasources import SOURCE_COLUMN_MAP
    assert isinstance(SOURCE_COLUMN_MAP, dict)
    required_keys = {"fpl_post_gw", "fpl_pre_gw", "understat", "espn",
                     "fpl_news", "premierinjuries", "ffs", "reddit"}
    assert required_keys.issubset(SOURCE_COLUMN_MAP.keys())


def test_source_column_map_fpl_post_gw_has_required_columns():
    from src.pipeline.datasources import SOURCE_COLUMN_MAP
    cols = SOURCE_COLUMN_MAP["fpl_post_gw"]["columns"]
    for col in ["minutes", "goals_scored", "assists", "expected_goals"]:
        assert col in cols
    assert "xg_chain" not in cols, "xg_chain must not be in fpl_post_gw — Understat owns it"


def test_source_column_map_understat_only_unique_cols():
    from src.pipeline.datasources import SOURCE_COLUMN_MAP
    cols = SOURCE_COLUMN_MAP["understat"]["columns"]
    assert cols == ["xg_chain", "xg_buildup"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd .worktrees/track-h
python -m pytest tests/datasources/test_signals.py::test_source_column_map_importable -v
```

Expected: `ImportError: cannot import name 'SOURCE_COLUMN_MAP'`

- [ ] **Step 3: Add SOURCE_COLUMN_MAP to `__init__.py`**

Replace the entire file with:

```python
from .signals import PlayerSignal, resolve_player_name, log_unresolved_name

SOURCE_COLUMN_MAP = {
    # --- Per-match actuals (post-GW). One row per PL match played. ---
    # Source: element-summary/{id}/history
    "fpl_post_gw": {
        "role": "primary",
        "competitions": ["PL"],
        "timing": "after_gw",
        "endpoint": "element-summary/{id}/history",
        "columns": [
            # Join / fixture context
            "element", "fixture", "round", "kickoff_time",
            "opponent_team", "was_home", "team_h_score", "team_a_score", "modified",
            # Performance
            "minutes", "starts",
            "goals_scored", "assists", "own_goals",
            "clean_sheets", "goals_conceded",
            "saves", "penalties_saved", "penalties_missed",
            "yellow_cards", "red_cards",
            # Opta xG / ICT
            "expected_goals", "expected_assists",
            "expected_goal_involvements", "expected_goals_conceded",
            "influence", "creativity", "threat", "ict_index",
            # Defensive
            "clearances_blocks_interceptions", "recoveries",
            "tackles", "defensive_contribution",
            # Bonus / points
            "bonus", "bps", "total_points",
            # Transfer market / ownership
            "transfers_in", "transfers_out", "transfers_balance", "selected",
            # Price at time of match (tenths of £M)
            "value",
        ],
    },

    # --- Pre-GW snapshot (available at prediction time). One row per player. ---
    # Source: bootstrap-static/elements
    # NOTE: season-to-date cumulative stats use same field names as fpl_post_gw
    # but represent season totals, not per-match rows.
    # Feature selection (Track C) decides which to promote to x-features.
    "fpl_pre_gw": {
        "role": "pre_gw_snapshot",
        "competitions": ["PL"],
        "timing": "before_gw",
        "endpoint": "bootstrap-static/elements",
        "columns": [
            # Identity / metadata
            "id", "code", "element_type", "team", "team_code",
            "first_name", "second_name", "web_name", "known_name",
            "opta_code", "squad_number", "birth_date", "region",
            "team_join_date", "removed", "special", "has_temporary_code", "photo",
            # Availability (feeds availability_features.py)
            "status", "news", "news_added",
            "chance_of_playing_this_round", "chance_of_playing_next_round",
            "can_transact", "can_select", "scout_risks", "scout_news_link",
            # Set piece role
            "corners_and_indirect_freekicks_order", "corners_and_indirect_freekicks_text",
            "direct_freekicks_order", "direct_freekicks_text",
            "penalties_order", "penalties_text",
            # Season-to-date cumulative stats (same field names as fpl_post_gw)
            "minutes", "starts",
            "goals_scored", "assists", "own_goals",
            "clean_sheets", "goals_conceded",
            "saves", "penalties_saved", "penalties_missed",
            "yellow_cards", "red_cards",
            "expected_goals", "expected_assists",
            "expected_goal_involvements", "expected_goals_conceded",
            "influence", "creativity", "threat", "ict_index",
            "clearances_blocks_interceptions", "recoveries",
            "tackles", "defensive_contribution",
            "bonus", "bps",
            # Per-90 season aggregates
            "expected_goals_per_90", "expected_assists_per_90",
            "expected_goal_involvements_per_90", "expected_goals_conceded_per_90",
            "clean_sheets_per_90", "saves_per_90",
            "goals_conceded_per_90", "starts_per_90", "defensive_contribution_per_90",
            # FPL form / prediction signals
            "form", "points_per_game", "ep_next", "ep_this",
            "event_points", "total_points",
            "dreamteam_count", "in_dreamteam",
            # Rank signals
            "ict_index_rank", "ict_index_rank_type",
            "creativity_rank", "creativity_rank_type",
            "threat_rank", "threat_rank_type",
            "influence_rank", "influence_rank_type",
            "now_cost_rank", "now_cost_rank_type",
            "form_rank", "form_rank_type",
            "points_per_game_rank", "points_per_game_rank_type",
            "selected_rank", "selected_rank_type",
            # Price / value
            "now_cost",
            "cost_change_event", "cost_change_event_fall",
            "cost_change_start", "cost_change_start_fall",
            "price_change_percent", "value_form", "value_season",
            # Transfer momentum
            "transfers_in", "transfers_out",
            "transfers_in_event", "transfers_out_event",
            # Ownership
            "selected_by_percent",
        ],
    },

    # --- Understat: unique creative chain metrics (PL only) ---
    # All other Understat columns (xg, xa, goals, etc.) overlap with FPL/Opta — dropped.
    "understat": {
        "role": "unique",
        "competitions": ["PL"],
        "timing": "after_gw",
        "client": "soccerdata.Understat (synchronous)",
        "columns": ["xg_chain", "xg_buildup"],
        "join_key": "(player_code, gw_date)",
    },

    # --- ESPN: all non-PL competitions ---
    # Confirmed slugs: uefa.champions, eng.fa, eng.league_cup, fifa.friendly
    # Unverified: uefa.europa, uefa.europa.conf — probe before use
    "espn": {
        "role": "primary",
        "competitions": ["UCL", "UEL", "UECL", "FA_Cup", "Carabao", "FIFA_Friendly", "INT"],
        "espn_league_slugs": [
            "uefa.champions",
            "uefa.europa",       # unverified — probe before use
            "uefa.europa.conf",  # unverified — probe before use
            "eng.fa",
            "eng.league_cup",
            "fifa.friendly",
        ],
        "timing": "after_match",
        "seasons": "2021-present",
        "endpoint": "sports.core.api.espn.com/v2/sports/soccer/athletes/{id}/eventlog",
        "columns": [
            "minutes", "goals", "assists",
            "shots", "shots_on_target",
            "yellow_cards", "red_cards",
            "fouls_committed", "fouls_suffered",
            "offsides",
        ],
    },

    # --- Availability signals ---
    "fpl_news": {
        "role": "availability_primary",
        "timing": "before_gw",
        "columns": ["is_injured", "is_doubt", "is_suspended", "availability_raw_text"],
    },
    "premierinjuries": {
        "role": "availability_fallback",
        "timing": "before_gw",
        "columns": ["is_injured", "is_doubt"],
    },
    "ffs": {
        "role": "availability_corroboration",
        "timing": "rolling",
        "columns": ["signal_type", "signal_confidence"],
    },
    "reddit": {
        "role": "availability_corroboration",
        "timing": "rolling",
        "columns": ["signal_type", "signal_confidence"],
    },
}

__all__ = ["PlayerSignal", "resolve_player_name", "log_unresolved_name", "SOURCE_COLUMN_MAP"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/datasources/test_signals.py -v
```

Expected: all pass including the 3 new SOURCE_COLUMN_MAP tests.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/datasources/__init__.py tests/datasources/test_signals.py
git commit -m "feat(H-C1): add SOURCE_COLUMN_MAP to datasources __init__"
```

---

## Task 2: Fix `understat.py` — replace async client + slim to unique columns

**Files:**
- Modify: `src/pipeline/datasources/understat.py`
- Modify: `tests/datasources/test_understat.py`

**Context on soccerdata.Understat output:**
- `sd.Understat(leagues="ENG-Premier League", seasons="2425")` — season format is `"2425"` for 2024-25, `"2324"` for 2023-24
- `.read_player_match_stats()` returns a MultiIndex DataFrame with index `['league', 'season', 'game', 'team', 'player']`
- The `game` index entry format: `"2024-08-16 Manchester United-Fulham"` — date is the first 10 chars
- Columns include `xg_chain`, `xg_buildup` (confirmed live) plus many overlapping columns (xg, xa, goals, etc. — all dropped)
- Join key: extract date from `game` index; use FPL fixtures API to map date → GW

**Context on date→GW mapping:**
- `fetch_gw_date_map(season_year)` calls `https://fantasy.premierleague.com/api/fixtures/?season={year}` for the season, returns `{date_str: gw_number}` dict where `date_str = kickoff_time[:10]`
- Blank GWs: a date may map to multiple GWs (DGW) — take the lower GW number as the first match date

- [ ] **Step 1: Write failing tests for the new understat interface**

Replace `tests/datasources/test_understat.py` entirely:

```python
"""Tests for the soccerdata.Understat-based understat client.

The module must:
1. Use soccerdata.Understat (synchronous) — NOT understatapi async
2. Return only xg_chain and xg_buildup columns
3. Join dates to GW numbers via a FPL fixtures map
4. Accept season format "2425" (for 2024-25), "2324" (for 2023-24)
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.pipeline.datasources.understat import (
    fetch_understat_xg_chain,
    build_date_gw_map,
    SEASON_FORMAT_EXAMPLES,
)


# Minimal MultiIndex DataFrame matching soccerdata.Understat output shape
def _mock_sd_df():
    idx = pd.MultiIndex.from_tuples(
        [
            ("ENG-Premier League", "2425", "2024-08-16 Arsenal-Wolves", "Arsenal", "Saka"),
            ("ENG-Premier League", "2425", "2024-08-16 Arsenal-Wolves", "Wolves", "Cunha"),
            ("ENG-Premier League", "2425", "2024-08-24 Arsenal-Brighton", "Arsenal", "Saka"),
        ],
        names=["league", "season", "game", "team", "player"],
    )
    return pd.DataFrame(
        {
            "xg": [0.45, 0.12, 0.55],
            "xg_chain": [0.60, 0.20, 0.70],
            "xg_buildup": [0.10, 0.05, 0.15],
            "xa": [0.10, 0.05, 0.12],
            "goals": [1, 0, 1],
            "assists": [0, 0, 1],
        },
        index=idx,
    )


MOCK_FIXTURES = [
    {"event": 1, "kickoff_time": "2024-08-16T19:30:00Z"},
    {"event": 2, "kickoff_time": "2024-08-24T15:00:00Z"},
]


def test_fetch_returns_only_xg_chain_and_xg_buildup():
    """Output must contain only xg_chain, xg_buildup (+ join keys)."""
    mock_sd = MagicMock()
    mock_sd.read_player_match_stats.return_value = _mock_sd_df()

    with patch("src.pipeline.datasources.understat._make_understat_reader", return_value=mock_sd):
        with patch("src.pipeline.datasources.understat._fetch_fixtures_for_season",
                   return_value=MOCK_FIXTURES):
            df = fetch_understat_xg_chain(season="2425")

    assert "xg_chain" in df.columns
    assert "xg_buildup" in df.columns
    # Must NOT contain overlapping FPL columns
    for col in ("xg", "xa", "goals", "assists", "shots", "key_passes",
                "yellow_cards", "red_cards"):
        assert col not in df.columns, f"Column '{col}' should be dropped (FPL overlap)"


def test_fetch_adds_gw_column():
    """Output must include a 'gw' column derived from match date."""
    mock_sd = MagicMock()
    mock_sd.read_player_match_stats.return_value = _mock_sd_df()

    with patch("src.pipeline.datasources.understat._make_understat_reader", return_value=mock_sd):
        with patch("src.pipeline.datasources.understat._fetch_fixtures_for_season",
                   return_value=MOCK_FIXTURES):
            df = fetch_understat_xg_chain(season="2425")

    assert "gw" in df.columns
    assert df[df["player"] == "Saka"]["gw"].iloc[0] == 1


def test_fetch_adds_player_and_team_columns():
    """Output must include player name and team from the MultiIndex."""
    mock_sd = MagicMock()
    mock_sd.read_player_match_stats.return_value = _mock_sd_df()

    with patch("src.pipeline.datasources.understat._make_understat_reader", return_value=mock_sd):
        with patch("src.pipeline.datasources.understat._fetch_fixtures_for_season",
                   return_value=MOCK_FIXTURES):
            df = fetch_understat_xg_chain(season="2425")

    assert "player" in df.columns
    assert "team" in df.columns


def test_build_date_gw_map_basic():
    """build_date_gw_map maps kickoff date strings to GW numbers."""
    gw_map = build_date_gw_map(MOCK_FIXTURES)
    assert gw_map["2024-08-16"] == 1
    assert gw_map["2024-08-24"] == 2


def test_build_date_gw_map_dgw_takes_lower_gw():
    """When two fixtures on same date have different GWs, take the lower GW."""
    fixtures = [
        {"event": 19, "kickoff_time": "2025-01-14T19:30:00Z"},
        {"event": 20, "kickoff_time": "2025-01-14T20:00:00Z"},
    ]
    gw_map = build_date_gw_map(fixtures)
    assert gw_map["2025-01-14"] == 19


def test_season_format_examples_exported():
    """SEASON_FORMAT_EXAMPLES must document the format convention."""
    assert isinstance(SEASON_FORMAT_EXAMPLES, dict)
    assert "2324" in SEASON_FORMAT_EXAMPLES
    assert "2425" in SEASON_FORMAT_EXAMPLES


def test_historical_season_logs_warning(caplog):
    """fetch_understat_xg_chain with a historical season must warn about GW mapping."""
    import logging
    mock_sd = MagicMock()
    mock_sd.read_player_match_stats.return_value = _mock_sd_df()

    with patch("src.pipeline.datasources.understat._make_understat_reader", return_value=mock_sd):
        with patch("src.pipeline.datasources.understat._fetch_fixtures_for_season",
                   return_value=MOCK_FIXTURES):
            with patch("src.pipeline.datasources.understat._current_understat_season",
                       return_value="2526"):  # current is 2526, so "2122" is historical
                with caplog.at_level(logging.WARNING, logger="src.pipeline.datasources.understat"):
                    fetch_understat_xg_chain(season="2122")

    assert any("historical" in r.message.lower() or "GW mapping" in r.message
               for r in caplog.records), "Expected warning about historical season GW mapping"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/datasources/test_understat.py -v
```

Expected: `ImportError` — `fetch_understat_xg_chain`, `build_date_gw_map`, `SEASON_FORMAT_EXAMPLES` not defined yet.

- [ ] **Step 3: Rewrite `understat.py`**

```python
"""Understat xg_chain / xg_buildup fetcher via soccerdata (synchronous).

Covers: PL only (ENG-Premier League).
Emits: xg_chain, xg_buildup — the only Understat columns with no FPL/Opta equivalent.
All other Understat columns (xg, xa, goals, assists, shots, key_passes,
yellow_cards, red_cards) overlap with FPL element-summary and are dropped.

Season format: "2425" for 2024-25, "2324" for 2023-24, "2223" for 2022-23, etc.
soccerdata caches data in ~/soccerdata/data/Understat/ on first fetch.

Join key: (player name, gw) — gw derived from match date via FPL fixtures API.
"""
from __future__ import annotations
import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)

# Season format: 4-char string concatenating last 2 digits of each year.
# e.g. 2023-24 → "2324", 2024-25 → "2425", 2025-26 → "2526"
SEASON_FORMAT_EXAMPLES = {
    "2324": "2023-24",
    "2425": "2024-25",
    "2526": "2025-26",
    "2223": "2022-23",
    "2122": "2021-22",
}

FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

# Columns soccerdata returns that overlap with FPL — must be dropped.
_OVERLAP_COLS = {
    "xg", "xa", "goals", "own_goals", "assists",
    "shots", "key_passes", "yellow_cards", "red_cards",
    "minutes", "position", "position_id",
    "league_id", "season_id", "game_id", "team_id", "player_id",
}


def _make_understat_reader(season: str):
    """Instantiate soccerdata.Understat reader. Extracted for testability."""
    import soccerdata as sd  # type: ignore[import]
    return sd.Understat(leagues="ENG-Premier League", seasons=season)


def _fetch_fixtures_for_season(season: str) -> list[dict]:
    """Fetch FPL fixture list for a season to build date→GW map.

    Args:
        season: 4-char understat season code e.g. "2425".

    Returns:
        List of fixture dicts with at minimum 'event' (GW) and 'kickoff_time'.

    KNOWN LIMITATION: The FPL fixtures endpoint returns current-season data only.
    For historical seasons (e.g. "2122"), the GW map will be built from current-season
    fixtures, causing almost all historical dates to map to null GW values.
    Full backfill support requires loading GW maps from archived bootstrap snapshots
    (results/snapshots/bootstrap_gw*.json) — deferred to Track C data pipeline build.
    """
    if season != _current_understat_season():
        logger.warning(
            "fetch_understat_xg_chain called with historical season '%s'. "
            "FPL fixtures endpoint returns current-season data only — "
            "GW mapping will be null for historical dates. "
            "Use bootstrap snapshot archives for historical backfill (Track C).",
            season,
        )
    resp = requests.get(FPL_FIXTURES_URL, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _current_understat_season() -> str:
    """Return the 4-char season code for the current/most recent season."""
    from datetime import date
    today = date.today()
    # Season rolls over in August
    if today.month >= 8:
        return f"{str(today.year)[2:]}{str(today.year + 1)[2:]}"
    return f"{str(today.year - 1)[2:]}{str(today.year)[2:]}"


def build_date_gw_map(fixtures: list[dict]) -> dict[str, int]:
    """Build a {date_str: gw_number} mapping from FPL fixtures.

    Date format: "YYYY-MM-DD" (first 10 chars of kickoff_time ISO string).
    DGW handling: if two fixtures share a date but different GW numbers,
    the lower GW number is used (first match of the DGW window).

    Args:
        fixtures: list of dicts with keys 'event' (int) and 'kickoff_time' (ISO str).

    Returns:
        dict mapping date string → GW number.
    """
    date_gw: dict[str, int] = {}
    for f in fixtures:
        if not f.get("kickoff_time") or not f.get("event"):
            continue
        date = f["kickoff_time"][:10]
        gw = int(f["event"])
        if date not in date_gw or gw < date_gw[date]:
            date_gw[date] = gw
    return date_gw


def fetch_understat_xg_chain(season: str = "2425") -> pd.DataFrame:
    """Return xg_chain and xg_buildup for all PL players in the given season.

    Args:
        season: 4-char season code (e.g. "2425" for 2024-25).

    Returns:
        DataFrame with columns: player, team, gw, xg_chain, xg_buildup
        Empty DataFrame (same columns) if soccerdata not installed or fetch fails.
    """
    empty = pd.DataFrame(columns=["player", "team", "gw", "xg_chain", "xg_buildup"])

    try:
        reader = _make_understat_reader(season)
    except ImportError:
        logger.warning("soccerdata not installed. pip install soccerdata")
        return empty
    except Exception as exc:
        logger.warning("Failed to init soccerdata.Understat: %s", exc)
        return empty

    try:
        raw = reader.read_player_match_stats()
    except Exception as exc:
        logger.warning("soccerdata.Understat.read_player_match_stats failed: %s", exc)
        return empty

    if raw.empty:
        return empty

    # Reset MultiIndex: league, season, game, team, player → columns
    df = raw.reset_index()

    # Extract date from game string e.g. "2024-08-16 Arsenal-Wolves" → "2024-08-16"
    df["match_date"] = df["game"].str[:10]

    # Build date → GW map
    try:
        fixtures = _fetch_fixtures_for_season(season)
        date_gw = build_date_gw_map(fixtures)
    except Exception as exc:
        logger.warning("Failed to build date→GW map: %s. GW column will be null.", exc)
        date_gw = {}

    df["gw"] = df["match_date"].map(date_gw)

    # Keep only unique columns + join keys; drop all FPL-overlapping columns
    keep = ["player", "team", "gw", "xg_chain", "xg_buildup"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        logger.warning("Expected columns missing from soccerdata output: %s", missing)
        return empty

    return df[keep].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/datasources/test_understat.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
python -m pytest tests/datasources/ -v
```

Expected: all existing tests still pass (understat tests previously patched `_fetch_player_grouped_stats_async` — now patched differently).

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/datasources/understat.py tests/datasources/test_understat.py
git commit -m "feat(H-C2): replace understatapi async with soccerdata.Understat sync; slim to xg_chain+xg_buildup"
```

---

## Task 3: Delete `soccerdata_client.py` + replace `test_soccerdata.py`

**Files:**
- Delete: `src/pipeline/datasources/soccerdata_client.py`
- Delete: `tests/datasources/test_soccerdata.py`

The FotMob wrapper is entirely replaced by `espn_client.py` (Task 4). The test file tests a deleted module — it must be removed before adding the ESPN tests.

- [ ] **Step 1: Verify test_soccerdata.py tests currently import the deleted module**

```bash
python -m pytest tests/datasources/test_soccerdata.py -v
```

Note the current passing tests (they mock `_fetch_fotmob_raw` internally).

- [ ] **Step 2: Delete both files**

Run from the worktree root (`.worktrees/track-h`):

```bash
cd .worktrees/track-h
git rm src/pipeline/datasources/soccerdata_client.py
git rm tests/datasources/test_soccerdata.py
```

- [ ] **Step 3: Verify no other file imports soccerdata_client**

```bash
grep -r "soccerdata_client" src/ tests/ --include="*.py"
```

Expected: no output. If any found, remove those imports.

- [ ] **Step 4: Run full test suite to confirm only the deleted tests are gone**

```bash
python -m pytest tests/datasources/ -v
```

Expected: remaining tests pass; `test_soccerdata.py` tests no longer appear.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(H-C3): delete soccerdata_client.py (FotMob) — replaced by espn_client"
```

---

## Task 4: Create `data/espn_player_id_map.csv`

**Files:**
- Create: `data/espn_player_id_map.csv`

The ID map seeds the ESPN client's ID resolver with manually verified FPL code → ESPN ID pairs for players who regularly appear in non-PL competitions. The fuzzy-match fallback handles unmapped players, but the seed prevents false matches for common-name players.

Columns: `fpl_code,web_name,espn_id,espn_name,verified`

- `fpl_code` — FPL persistent player code (from `bootstrap/elements[].code`)
- `web_name` — FPL display name (for human readability)
- `espn_id` — ESPN athlete numeric ID
- `espn_name` — ESPN display name (may differ from FPL)
- `verified` — `true` if confirmed via live eventlog pull; `false` if fuzzy-inferred

- [ ] **Step 1: Create the CSV with confirmed entries**

```
fpl_code,web_name,espn_id,espn_name,verified
448047,Enzo Fernández,285450,Enzo Fernandez,true
244723,Palmer,296395,Cole Palmer,true
```

Save to `data/espn_player_id_map.csv`.

Note: This seed will grow over time. The ESPN client (Task 5) appends fuzzy-matched entries with `verified=false` to this file when it resolves new players.

- [ ] **Step 2: Verify file is readable**

```bash
python -c "import pandas as pd; df = pd.read_csv('data/espn_player_id_map.csv'); print(df.dtypes); print(df)"
```

Expected: 2 rows, columns as specified, `verified` reads as bool or object.

- [ ] **Step 3: Commit**

```bash
git add data/espn_player_id_map.csv
git commit -m "data(H-C4): seed espn_player_id_map.csv with confirmed Enzo/Palmer ESPN IDs"
```

---

## Task 5: Create `espn_client.py`

**Files:**
- Create: `src/pipeline/datasources/espn_client.py`
- Create: `tests/datasources/test_espn_client.py`

**ESPN API context (confirmed from live testing):**
- Eventlog endpoint: `https://sports.core.api.espn.com/v2/sports/soccer/athletes/{espn_id}/eventlog`
  Returns `{"season": {...}, "events": {"items": [{"$ref": "..."}]}}` where each `$ref` URL contains the league slug.
- Match summary endpoint: the `$ref` URL from eventlog resolves to an event, which links to a `competitors` array with `statistics` per player.
- PL filter: skip events where the league slug is `eng.1`.
- Rate limiting: 1.0s sleep between player fetches; exponential backoff on 429 (2s, 4s, 8s, max 3 retries).
- Cache: `results/espn_cache/player_{espn_id}_season_{year}.csv` — if exists, skip fetch.
- Unresolved IDs: log to `results/espn_unresolved.csv`.

**Fuzzy match for ID resolution:**
- ESPN team roster: `https://sports.core.api.espn.com/v2/sports/soccer/leagues/eng.1/seasons/{year}/teams/{team_espn_id}/roster`
- Use `difflib.SequenceMatcher` to match `web_name` against ESPN roster names.
- Threshold: ≥ 0.85 similarity ratio.

- [ ] **Step 1: Write failing tests**

Create `tests/datasources/test_espn_client.py`:

```python
"""Tests for espn_client.py — ESPN eventlog fetch for non-PL minutes.

All HTTP calls are mocked. No real network requests in tests.
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.pipeline.datasources.espn_client import (
    resolve_espn_player_id,
    fetch_espn_player_season,
    fetch_espn_recent,
    NON_PL_SLUGS,
    PL_SLUG,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_ID_MAP = pd.DataFrame([
    {"fpl_code": 448047, "web_name": "Enzo Fernández",
     "espn_id": 285450, "espn_name": "Enzo Fernandez", "verified": True},
    {"fpl_code": 244723, "web_name": "Palmer",
     "espn_id": 296395, "espn_name": "Cole Palmer", "verified": True},
])

# Minimal eventlog API response — one UCL event, one PL event (should be filtered)
MOCK_EVENTLOG = {
    "events": {
        "items": [
            {"$ref": "https://sports.core.api.espn.com/v2/sports/soccer/leagues/uefa.champions/events/12345"},
            {"$ref": "https://sports.core.api.espn.com/v2/sports/soccer/leagues/eng.1/events/99999"},
        ]
    }
}

# Minimal event response for the UCL event
MOCK_UCL_EVENT = {
    "id": "12345",
    "date": "2025-03-05T20:00:00Z",
    "competitions": [
        {
            "competitors": [
                {
                    "id": "285450",
                    "statistics": [
                        {"name": "minutesPlayed", "value": 87.0},
                        {"name": "goals", "value": 1.0},
                        {"name": "assists", "value": 0.0},
                        {"name": "shots", "value": 3.0},
                        {"name": "shotsOnTarget", "value": 2.0},
                        {"name": "yellowCards", "value": 0.0},
                        {"name": "redCards", "value": 0.0},
                        {"name": "foulsCommitted", "value": 1.0},
                        {"name": "foulsSuffered", "value": 2.0},
                        {"name": "offsides", "value": 0.0},
                    ],
                }
            ]
        }
    ],
}


# ── ID resolution tests ───────────────────────────────────────────────────────

def test_resolve_from_seed_map(tmp_path):
    """resolve_espn_player_id returns known ESPN ID from seeded CSV."""
    with patch("src.pipeline.datasources.espn_client._load_id_map", return_value=MOCK_ID_MAP):
        espn_id = resolve_espn_player_id(
            fpl_code=448047, web_name="Enzo Fernández", second_name="Fernández"
        )
    assert espn_id == 285450


def test_resolve_returns_none_for_unknown(tmp_path):
    """resolve_espn_player_id returns None if player not in map and fuzzy fails."""
    with patch("src.pipeline.datasources.espn_client._load_id_map", return_value=MOCK_ID_MAP):
        with patch("src.pipeline.datasources.espn_client._fuzzy_resolve_espn_id",
                   return_value=None):
            espn_id = resolve_espn_player_id(
                fpl_code=999999, web_name="Unknown Player", second_name="Player"
            )
    assert espn_id is None


# ── Season fetch tests ────────────────────────────────────────────────────────

def test_fetch_espn_player_season_returns_dataframe():
    """fetch_espn_player_season returns a DataFrame with required columns."""
    mock_get = MagicMock()
    mock_get.side_effect = [
        MagicMock(json=lambda: MOCK_EVENTLOG, raise_for_status=lambda: None),
        MagicMock(json=lambda: MOCK_UCL_EVENT, raise_for_status=lambda: None),
        # PL event is filtered before fetch — only 1 event response needed
    ]
    with patch("src.pipeline.datasources.espn_client.requests.get", mock_get):
        with patch("src.pipeline.datasources.espn_client.time.sleep"):
            df = fetch_espn_player_season(espn_id=285450, season_year=2025)

    assert isinstance(df, pd.DataFrame)
    assert "minutes" in df.columns
    assert "goals" in df.columns
    assert "competition_slug" in df.columns


def test_fetch_espn_player_season_filters_pl():
    """fetch_espn_player_season must NOT include Premier League (eng.1) events."""
    mock_get = MagicMock()
    mock_get.side_effect = [
        MagicMock(json=lambda: MOCK_EVENTLOG, raise_for_status=lambda: None),
        MagicMock(json=lambda: MOCK_UCL_EVENT, raise_for_status=lambda: None),
    ]
    with patch("src.pipeline.datasources.espn_client.requests.get", mock_get):
        with patch("src.pipeline.datasources.espn_client.time.sleep"):
            df = fetch_espn_player_season(espn_id=285450, season_year=2025)

    # PL slug must be absent
    if not df.empty:
        assert PL_SLUG not in df["competition_slug"].values


def test_fetch_espn_player_season_uses_cache(tmp_path):
    """fetch_espn_player_season skips HTTP if cache file exists."""
    cache_file = tmp_path / "player_285450_season_2025.csv"
    cached_df = pd.DataFrame([{
        "espn_id": 285450, "match_date": "2025-03-05",
        "competition_slug": "uefa.champions",
        "minutes": 90, "goals": 1, "assists": 0,
        "shots": 2, "shots_on_target": 1,
        "yellow_cards": 0, "red_cards": 0,
        "fouls_committed": 0, "fouls_suffered": 1, "offsides": 0,
    }])
    cached_df.to_csv(cache_file, index=False)

    with patch("src.pipeline.datasources.espn_client._get_cache_path",
               return_value=cache_file):
        with patch("src.pipeline.datasources.espn_client.requests.get") as mock_get:
            df = fetch_espn_player_season(espn_id=285450, season_year=2025)
            mock_get.assert_not_called()  # HTTP must not be called

    assert len(df) == 1
    assert df.iloc[0]["goals"] == 1


def test_fetch_espn_player_season_empty_on_http_error():
    """fetch_espn_player_season returns empty DataFrame on HTTP failure."""
    with patch("src.pipeline.datasources.espn_client.requests.get",
               side_effect=Exception("network error")):
        with patch("src.pipeline.datasources.espn_client.time.sleep"):
            df = fetch_espn_player_season(espn_id=285450, season_year=2025)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_fetch_espn_player_season_skips_event_with_missing_date():
    """Events with empty/missing date field must be skipped, not included with date=''."""
    event_missing_date = dict(MOCK_UCL_EVENT)
    event_missing_date.pop("date", None)  # remove date entirely

    mock_get = MagicMock()
    mock_get.side_effect = [
        MagicMock(json=lambda: MOCK_EVENTLOG, raise_for_status=lambda: None),
        MagicMock(json=lambda: event_missing_date, raise_for_status=lambda: None),
    ]
    with patch("src.pipeline.datasources.espn_client.requests.get", mock_get):
        with patch("src.pipeline.datasources.espn_client.time.sleep"):
            df = fetch_espn_player_season(espn_id=285450, season_year=2025)

    # The event with no date must not produce a row with match_date=""
    if not df.empty:
        assert "" not in df["match_date"].values


# ── Recent fetch tests ────────────────────────────────────────────────────────

def test_fetch_espn_recent_returns_last_n_days():
    """fetch_espn_recent returns rows within the last `days` window."""
    rows = [
        {"espn_id": 285450, "match_date": "2025-03-20",
         "competition_slug": "uefa.champions", "minutes": 90,
         "goals": 0, "assists": 1, "shots": 1, "shots_on_target": 0,
         "yellow_cards": 0, "red_cards": 0,
         "fouls_committed": 0, "fouls_suffered": 0, "offsides": 0},
        {"espn_id": 285450, "match_date": "2024-12-01",  # old — should be filtered
         "competition_slug": "fifa.friendly", "minutes": 70,
         "goals": 1, "assists": 0, "shots": 2, "shots_on_target": 1,
         "yellow_cards": 0, "red_cards": 0,
         "fouls_committed": 1, "fouls_suffered": 0, "offsides": 0},
    ]
    full_season = pd.DataFrame(rows)

    with patch("src.pipeline.datasources.espn_client.fetch_espn_player_season",
               return_value=full_season):
        df = fetch_espn_recent(espn_id=285450, season_year=2025, days=30,
                               reference_date="2025-04-05")

    assert len(df) == 1
    assert df.iloc[0]["competition_slug"] == "uefa.champions"


# ── Constants ─────────────────────────────────────────────────────────────────

def test_pl_slug_is_excluded():
    """PL_SLUG must be 'eng.1' and must not appear in NON_PL_SLUGS."""
    assert PL_SLUG == "eng.1"
    assert PL_SLUG not in NON_PL_SLUGS


def test_non_pl_slugs_contains_confirmed_competitions():
    confirmed = {"uefa.champions", "eng.fa", "eng.league_cup", "fifa.friendly"}
    assert confirmed.issubset(NON_PL_SLUGS)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/datasources/test_espn_client.py -v
```

Expected: `ImportError: cannot import name 'resolve_espn_player_id' from 'src.pipeline.datasources.espn_client'`

- [ ] **Step 3: Create `espn_client.py`**

```python
"""ESPN eventlog client — non-PL competition minutes and match stats.

Covers: UCL, UEL, UECL, FA Cup (eng.fa), Carabao Cup (eng.league_cup),
        FIFA Friendly (fifa.friendly).
Does NOT cover: Premier League (eng.1) — use FPL API element-summary instead.
No xG/xA available from ESPN (confirmed via live testing).

ESPN API endpoints (no auth required, public):
  Eventlog:  https://sports.core.api.espn.com/v2/sports/soccer/athletes/{id}/eventlog
  Event:     URL embedded in eventlog $ref items

Rate limiting: 1.0s sleep between player requests; exponential backoff on 429.
Cache: results/espn_cache/player_{espn_id}_season_{year}.csv (idempotent re-runs).
ID map: data/espn_player_id_map.csv — fpl_code,web_name,espn_id,espn_name,verified
"""
from __future__ import annotations
import difflib
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.config import BASE_DIR

logger = logging.getLogger(__name__)

# Premier League slug — always excluded from non-PL fetch
PL_SLUG = "eng.1"

# Non-PL competition slugs to include.
# uefa.europa and uefa.europa.conf are unverified — probe before use in production.
NON_PL_SLUGS: set[str] = {
    "uefa.champions",
    "uefa.europa",       # unverified slug — probe before use
    "uefa.europa.conf",  # unverified slug — probe before use
    "eng.fa",
    "eng.league_cup",
    "fifa.friendly",
}

ESPN_EVENTLOG_URL = (
    "https://sports.core.api.espn.com/v2/sports/soccer/athletes/{espn_id}/eventlog"
)
ID_MAP_PATH = BASE_DIR / "data" / "espn_player_id_map.csv"
CACHE_DIR = BASE_DIR / "results" / "espn_cache"
UNRESOLVED_PATH = BASE_DIR / "results" / "espn_unresolved.csv"

# Stat name mapping: ESPN statistic name → output column name
_STAT_MAP = {
    "minutesPlayed": "minutes",
    "goals": "goals",
    "assists": "assists",
    "shots": "shots",
    "shotsOnTarget": "shots_on_target",
    "yellowCards": "yellow_cards",
    "redCards": "red_cards",
    "foulsCommitted": "fouls_committed",
    "foulsSuffered": "fouls_suffered",
    "offsides": "offsides",
}

_OUTPUT_COLS = [
    "espn_id", "match_date", "competition_slug",
    "minutes", "goals", "assists", "shots", "shots_on_target",
    "yellow_cards", "red_cards", "fouls_committed", "fouls_suffered", "offsides",
]


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _get_cache_path(espn_id: int, season_year: int) -> Path:
    return CACHE_DIR / f"player_{espn_id}_season_{season_year}.csv"


# ── ID map helpers ────────────────────────────────────────────────────────────

def _load_id_map() -> pd.DataFrame:
    if ID_MAP_PATH.exists():
        return pd.read_csv(ID_MAP_PATH)
    return pd.DataFrame(columns=["fpl_code", "web_name", "espn_id", "espn_name", "verified"])


def _log_unresolved(fpl_code: int, web_name: str) -> None:
    row = pd.DataFrame([{"fpl_code": fpl_code, "web_name": web_name}])
    if UNRESOLVED_PATH.exists():
        row.to_csv(UNRESOLVED_PATH, mode="a", header=False, index=False)
    else:
        UNRESOLVED_PATH.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(UNRESOLVED_PATH, index=False)


def _fuzzy_resolve_espn_id(
    web_name: str,
    second_name: str,
    threshold: float = 0.85,
) -> int | None:
    """Attempt fuzzy name match against ESPN roster. Returns ESPN ID or None."""
    # Fuzzy matching requires knowing ESPN team roster URLs — not implemented here.
    # Placeholder: always returns None. In production, iterate known ESPN team
    # roster endpoints and fuzzy-match player names.
    logger.debug("Fuzzy resolve not implemented for '%s'", web_name)
    return None


def resolve_espn_player_id(
    fpl_code: int,
    web_name: str,
    second_name: str,
) -> int | None:
    """Resolve an FPL player code to an ESPN athlete ID.

    Resolution order:
    1. Exact fpl_code match in data/espn_player_id_map.csv
    2. Fuzzy name match with ≥0.85 confidence (if roster data available)
    3. Log to results/espn_unresolved.csv and return None

    Args:
        fpl_code: FPL persistent player code (element[].code).
        web_name: FPL web_name (e.g. "Enzo Fernández").
        second_name: FPL second_name (used as fallback for fuzzy match).

    Returns:
        ESPN athlete integer ID, or None if unresolvable.
    """
    id_map = _load_id_map()
    match = id_map[id_map["fpl_code"] == fpl_code]
    if not match.empty:
        return int(match.iloc[0]["espn_id"])

    # Fuzzy fallback
    espn_id = _fuzzy_resolve_espn_id(web_name, second_name)
    if espn_id is not None:
        return espn_id

    _log_unresolved(fpl_code, web_name)
    return None


# ── Fetch helpers ─────────────────────────────────────────────────────────────

def _extract_slug_from_ref(ref_url: str) -> str | None:
    """Extract league slug from an ESPN $ref URL.

    URL pattern: .../leagues/{slug}/events/{id}
    Returns slug string or None if pattern not matched.
    """
    parts = ref_url.split("/")
    try:
        leagues_idx = parts.index("leagues")
        return parts[leagues_idx + 1]
    except (ValueError, IndexError):
        return None


def _fetch_with_retry(url: str, max_retries: int = 3) -> dict:
    """GET url with exponential backoff on 429. Raises on non-recoverable errors."""
    delay = 2.0
    for attempt in range(max_retries + 1):
        resp = requests.get(url, timeout=15)
        if resp.status_code == 429:
            if attempt < max_retries:
                logger.warning("ESPN 429 rate limit — waiting %.0fs", delay)
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
        resp.raise_for_status()
        return resp.json()
    return {}  # unreachable


def _parse_match_stats(event_data: dict, espn_id: int) -> dict | None:
    """Extract per-player stats from an ESPN event response.

    Finds the competitor entry matching espn_id and maps ESPN stat names
    to output column names. Returns None if player not found in event.
    """
    for competition in event_data.get("competitions", []):
        for competitor in competition.get("competitors", []):
            if str(competitor.get("id")) == str(espn_id):
                stats: dict[str, float] = {}
                for stat in competitor.get("statistics", []):
                    col = _STAT_MAP.get(stat["name"])
                    if col:
                        stats[col] = float(stat.get("value", 0))
                return stats
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_espn_player_season(
    espn_id: int,
    season_year: int,
) -> pd.DataFrame:
    """Return all non-PL match stats for a player in a given season.

    Uses cache if available (results/espn_cache/player_{id}_season_{year}.csv).
    Filters out Premier League events (eng.1 slug).
    Sleeps 1s between event fetches for rate limiting.

    Args:
        espn_id: ESPN athlete numeric ID.
        season_year: Season end year (e.g. 2025 for 2024-25).

    Returns:
        DataFrame with columns matching _OUTPUT_COLS.
        Empty DataFrame on any unrecoverable error.
    """
    empty = pd.DataFrame(columns=_OUTPUT_COLS)
    cache_path = _get_cache_path(espn_id, season_year)

    if cache_path.exists():
        logger.debug("ESPN cache hit: %s", cache_path)
        return pd.read_csv(cache_path)

    try:
        eventlog = _fetch_with_retry(
            ESPN_EVENTLOG_URL.format(espn_id=espn_id)
        )
    except Exception as exc:
        logger.warning("ESPN eventlog fetch failed for espn_id=%s: %s", espn_id, exc)
        return empty

    items = eventlog.get("events", {}).get("items", [])
    rows = []

    for item in items:
        ref_url = item.get("$ref", "")
        slug = _extract_slug_from_ref(ref_url)
        if not slug or slug == PL_SLUG:
            continue  # skip PL events
        if slug not in NON_PL_SLUGS:
            continue  # skip unrecognised competitions

        time.sleep(1.0)
        try:
            event_data = _fetch_with_retry(ref_url)
        except Exception as exc:
            logger.warning("ESPN event fetch failed (%s): %s", ref_url, exc)
            continue

        match_date = event_data.get("date", "")[:10]
        if not match_date:
            logger.warning("ESPN event missing date field, skipping (%s)", ref_url)
            continue
        stats = _parse_match_stats(event_data, espn_id)
        if stats is None:
            continue

        rows.append({
            "espn_id": espn_id,
            "match_date": match_date,
            "competition_slug": slug,
            **{col: stats.get(col, 0) for col in _OUTPUT_COLS
               if col not in ("espn_id", "match_date", "competition_slug")},
        })

    if not rows:
        return empty

    df = pd.DataFrame(rows)[_OUTPUT_COLS]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def fetch_espn_recent(
    espn_id: int,
    season_year: int,
    days: int = 30,
    reference_date: str | None = None,
) -> pd.DataFrame:
    """Return non-PL match stats from the last `days` days.

    Used in the weekly prediction run to compute fatigue signal.

    Args:
        espn_id: ESPN athlete numeric ID.
        season_year: Season end year for the fetch.
        days: Window size in days (default 30).
        reference_date: ISO date string "YYYY-MM-DD". Defaults to today.

    Returns:
        Subset of fetch_espn_player_season output within the date window.
    """
    ref = datetime.strptime(reference_date, "%Y-%m-%d") if reference_date \
        else datetime.today()
    cutoff = (ref - timedelta(days=days)).strftime("%Y-%m-%d")

    df = fetch_espn_player_season(espn_id=espn_id, season_year=season_year)
    if df.empty:
        return df

    return df[df["match_date"] >= cutoff].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/datasources/test_espn_client.py -v
```

Expected: all 10 tests pass.

- [ ] **Step 5: Run full datasources test suite**

```bash
python -m pytest tests/datasources/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/datasources/espn_client.py tests/datasources/test_espn_client.py
git commit -m "feat(H-C5): add espn_client.py — non-PL eventlog fetch, caching, ID resolver"
```

---

## Task 6: Create `availability_features.py`

**Files:**
- Create: `src/pipeline/datasources/availability_features.py`
- Create: `tests/datasources/test_availability_features.py`

**Context on existing availability.py:**
- Lives at `src/pipeline/availability.py`
- `filter_availability(predictions_df, bootstrap_data)` → hard-excludes and soft-scales xP
- Hard excludes: status in `{i, u, s, n}` or chance in `{0, 25}`
- Soft scales: chance 50 → 0.5×, chance 75 → 0.75×, status d + null chance → 0.5×
- This module is NOT modified. `availability_features.py` is a new wrapper.

**What `availability_features.py` adds:**
The existing `availability.py` does live xP scaling for the optimizer. The new module produces *feature columns* for the ML training dataset — different purpose. It generates `is_injured`, `is_doubt`, `is_suspended`, `signal_confidence`, `n_corroborating_sources` for every player in a bootstrap snapshot.

**Status→feature mapping:**
- `is_injured = 1` if status in `{i, u}` (physically unavailable due to injury)
- `is_suspended = 1` if status == `s`
- `is_doubt = 1` if status == `d` (NOT chance threshold — see spec)
- `status == n` → player excluded from dataset entirely, not flagged
- Fallback: if status == `a` AND news == `""` → check premierinjuries signals

- [ ] **Step 1: Write failing tests**

Create `tests/datasources/test_availability_features.py`:

```python
"""Tests for availability_features.py — ML feature column assembly.

Tests cover all FPL status values, the premierinjuries fallback path,
corroboration counting, and the is_doubt edge case (status=d + chance=100).
"""
import pytest
from src.pipeline.datasources.availability_features import (
    compute_availability_features,
    AvailabilityFeatures,
)
from src.pipeline.datasources.signals import PlayerSignal


def _fpl_element(status: str, news: str = "", chance: int | None = None) -> dict:
    return {
        "id": 237,
        "code": 448047,
        "web_name": "Enzo Fernández",
        "status": status,
        "news": news,
        "news_added": "2026-04-05T10:00:00Z" if news else None,
        "chance_of_playing_next_round": chance,
    }


# ── Status → is_injured ────────────────────────────────────────────────────────

def test_status_i_sets_is_injured():
    result = compute_availability_features(_fpl_element("i", news="Hamstring"), [])
    assert result.is_injured == 1
    assert result.is_suspended == 0
    assert result.is_doubt == 0


def test_status_u_sets_is_injured():
    result = compute_availability_features(_fpl_element("u", news="Long-term knee"), [])
    assert result.is_injured == 1


def test_status_s_sets_is_suspended():
    result = compute_availability_features(_fpl_element("s", news="3-match ban"), [])
    assert result.is_injured == 0
    assert result.is_suspended == 1
    assert result.is_doubt == 0


def test_status_d_sets_is_doubt():
    result = compute_availability_features(_fpl_element("d", news="Hamstring 75%"), [])
    assert result.is_doubt == 1
    assert result.is_injured == 0


def test_status_a_all_zeros():
    result = compute_availability_features(_fpl_element("a"), [])
    assert result.is_injured == 0
    assert result.is_doubt == 0
    assert result.is_suspended == 0


# ── is_doubt edge case: status=d + chance=100 ──────────────────────────────────

def test_doubt_with_chance_100_still_sets_is_doubt():
    """status=d + chance_of_playing=100 → is_doubt=1 (status wins, not chance)."""
    result = compute_availability_features(_fpl_element("d", news="", chance=100), [])
    assert result.is_doubt == 1


# ── Premierinjuries fallback ───────────────────────────────────────────────────

def test_premierinjuries_fallback_when_fpl_clear():
    """If FPL status=a and news='', premierinjuries signal overrides."""
    pi_signal = PlayerSignal(
        player_code=448047,
        source="premierinjuries",
        signal_type="injured",
        text="Out — hamstring",
        timestamp="2026-04-05T08:00:00Z",
        confidence=0.8,
    )
    result = compute_availability_features(_fpl_element("a", news=""), [pi_signal])
    assert result.is_injured == 1


def test_premierinjuries_fallback_not_used_when_fpl_has_news():
    """If FPL has news, premierinjuries fallback must NOT be used."""
    pi_signal = PlayerSignal(
        player_code=448047,
        source="premierinjuries",
        signal_type="injured",
        text="Out — hamstring",
        timestamp="2026-04-05T08:00:00Z",
        confidence=0.8,
    )
    # FPL says available with news — trust FPL
    result = compute_availability_features(
        _fpl_element("a", news="Slight knock, should be fine"), [pi_signal]
    )
    assert result.is_injured == 0


# ── Corroboration counting ─────────────────────────────────────────────────────

def test_n_corroborating_sources_with_agreement():
    """Two agreeing signals → n_corroborating_sources=2."""
    ffs_signal = PlayerSignal(
        player_code=448047, source="ffs", signal_type="doubt",
        text="50/50 this week", timestamp="2026-04-05T09:00:00Z", confidence=0.6,
    )
    reddit_signal = PlayerSignal(
        player_code=448047, source="reddit", signal_type="doubt",
        text="Not in training", timestamp="2026-04-05T07:00:00Z", confidence=0.5,
    )
    result = compute_availability_features(
        _fpl_element("d", news="Knock — 75%"), [ffs_signal, reddit_signal]
    )
    assert result.n_corroborating_sources == 2


def test_signal_confidence_fpl_only():
    """With no external signals, signal_confidence should equal FPL weight (1.0)."""
    result = compute_availability_features(_fpl_element("i", news="Knee injury"), [])
    assert result.signal_confidence == pytest.approx(1.0)


def test_signal_confidence_increases_with_corroboration():
    """Adding corroborating signals should increase or maintain signal_confidence."""
    baseline = compute_availability_features(
        _fpl_element("d", news="50/50"), []
    )
    pi_signal = PlayerSignal(
        player_code=448047, source="premierinjuries", signal_type="doubt",
        text="50/50", timestamp="2026-04-05T08:00:00Z", confidence=0.8,
    )
    with_pi = compute_availability_features(
        _fpl_element("d", news="50/50"), [pi_signal]
    )
    assert with_pi.signal_confidence >= baseline.signal_confidence


# ── Return type ────────────────────────────────────────────────────────────────

def test_returns_availability_features_dataclass():
    result = compute_availability_features(_fpl_element("a"), [])
    assert isinstance(result, AvailabilityFeatures)
    assert hasattr(result, "is_injured")
    assert hasattr(result, "is_doubt")
    assert hasattr(result, "is_suspended")
    assert hasattr(result, "signal_confidence")
    assert hasattr(result, "n_corroborating_sources")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/datasources/test_availability_features.py -v
```

Expected: `ImportError: cannot import name 'compute_availability_features'`

- [ ] **Step 3: Create `availability_features.py`**

```python
"""Unified availability feature assembly for ML training dataset.

Produces four feature columns per player from FPL status + corroborating signals:
  is_injured            — FPL status in {i, u}
  is_suspended          — FPL status == s
  is_doubt              — FPL status == d (NOT chance threshold)
  signal_confidence     — weighted average of agreeing source confidences
  n_corroborating_sources — count of external signals agreeing with primary

This module is for FEATURE GENERATION (ML dataset building), not xP scaling.
For live xP scaling in the optimizer, see src/pipeline/availability.py.

Relationship to availability.py:
  availability.py.filter_availability() → called by predict.py for live xP scaling
  availability_features.py              → called during training data preparation

Status → feature mapping:
  i (injured)         → is_injured=1
  u (unavailable)     → is_injured=1  (physically out, not injury but same impact)
  s (suspended)       → is_suspended=1 (separate column — different xP impact)
  d (doubtful)        → is_doubt=1    (driven by status, NOT by chance threshold)
  a (available)       → all zeros (unless premierinjuries fallback fires)
  n (not in squad)    → player excluded from dataset, never reaches this module

Signal weights (for confidence aggregation):
  FPL primary:       1.0
  premierinjuries:   0.8
  FFS RSS:           0.6
  Reddit:            0.5
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

from src.pipeline.datasources.signals import PlayerSignal

logger = logging.getLogger(__name__)

# Source confidence weights
_SOURCE_WEIGHTS: dict[str, float] = {
    "fpl":             1.0,
    "premierinjuries": 0.8,
    "ffs":             0.6,
    "reddit":          0.5,
}

# FPL status codes
_INJURED_STATUSES = {"i", "u"}
_SUSPENDED_STATUS = "s"
_DOUBT_STATUS = "d"


@dataclass
class AvailabilityFeatures:
    """Availability feature columns for one player at a given GW snapshot."""
    is_injured: int               # 1 if status in {i, u}
    is_suspended: int             # 1 if status == s
    is_doubt: int                 # 1 if status == d
    signal_confidence: float      # weighted average of agreeing source confidences
    n_corroborating_sources: int  # count of external signals agreeing with primary


def _compute_weighted_confidence(
    primary_weight: float,
    agreeing_signals: list[PlayerSignal],
) -> float:
    """Weighted average confidence across primary + agreeing sources.

    If no external signals, returns primary_weight unchanged (typically 1.0 for FPL).
    """
    weights = [primary_weight]
    for sig in agreeing_signals:
        w = _SOURCE_WEIGHTS.get(sig.source, 0.5)
        weights.append(w)
    return sum(weights) / len(weights)


def compute_availability_features(
    element: dict,
    external_signals: list[PlayerSignal],
) -> AvailabilityFeatures:
    """Derive availability feature columns for one player.

    Args:
        element: FPL bootstrap element dict with at minimum:
            status, news, news_added, chance_of_playing_next_round
        external_signals: list of PlayerSignal from premierinjuries, FFS, Reddit.
            Only signals for this player's code should be passed in.

    Returns:
        AvailabilityFeatures dataclass.
    """
    status = element.get("status", "a")
    news = element.get("news", "") or ""

    # Primary flags from FPL status
    is_injured = int(status in _INJURED_STATUSES)
    is_suspended = int(status == _SUSPENDED_STATUS)
    is_doubt = int(status == _DOUBT_STATUS)

    # Premierinjuries fallback: only if FPL is fully clear (status=a, no news)
    pi_signals = [s for s in external_signals if s.source == "premierinjuries"]
    if status == "a" and not news and pi_signals:
        pi = pi_signals[0]
        if pi.signal_type == "injured":
            is_injured = 1
        elif pi.signal_type == "doubt":
            is_doubt = 1

    # Determine primary signal type for corroboration matching
    if is_injured:
        primary_type = "injured"
    elif is_suspended:
        primary_type = "suspended"
    elif is_doubt:
        primary_type = "doubt"
    else:
        primary_type = "available"

    # Corroborating signals: those agreeing with the primary signal type
    # For "available", corroboration means any source also says available
    corroborating = [
        s for s in external_signals
        if s.signal_type == primary_type or
        (primary_type == "injured" and s.signal_type in {"injured", "doubt"})
    ]

    signal_confidence = _compute_weighted_confidence(
        primary_weight=_SOURCE_WEIGHTS["fpl"],
        agreeing_signals=corroborating,
    )
    n_corroborating = len(corroborating)

    return AvailabilityFeatures(
        is_injured=is_injured,
        is_suspended=is_suspended,
        is_doubt=is_doubt,
        signal_confidence=signal_confidence,
        n_corroborating_sources=n_corroborating,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/datasources/test_availability_features.py -v
```

Expected: all 13 tests pass.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all tests pass, no regressions in main pipeline tests.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/datasources/availability_features.py tests/datasources/test_availability_features.py
git commit -m "feat(H-C6): add availability_features.py — ML feature columns with is_injured, is_doubt, is_suspended, signal_confidence"
```

---

## Task 7: Update `source_validation.py` docstring

**Files:**
- Modify: `src/pipeline/source_validation.py` (docstring only — no logic changes)

The module docstring is out of date: it describes the gate as validating xG vs actual goals. Now that xG is owned by FPL/Opta, the gate validates `xg_chain`/`xg_buildup` correlation against actual goal-chain outcomes. Also confirm in docstring that the file lives at pipeline root, not inside datasources.

- [ ] **Step 1: Update the module docstring**

In `src/pipeline/source_validation.py`, replace the module docstring (lines 1–9) with:

```python
"""xG chain source validation gate for Track H / Track B dependency.

Gate rule: understat_rho >= fpl_opta_rho - tolerance → use understat xg_chain/xg_buildup.
If gate fails → fall back to vaastav goals_conceded for xGC_rolling_4.

Note: This module lives at src/pipeline/ (pipeline root), NOT inside datasources/.
It is a pipeline-level gate, not a per-source module.

Since xG and xA are now sourced from FPL/Opta (element-summary), the Spearman ρ gate
validates xg_chain / xg_buildup correlation against actual goal-chain outcomes —
not xG vs actual goals.

Usage:
    python -m src.pipeline.source_validation
"""
```

- [ ] **Step 2: Run existing source_validation tests to confirm no regressions**

```bash
python -m pytest tests/datasources/test_source_validation.py -v
```

Expected: all 4 existing tests pass (no logic was changed).

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/source_validation.py
git commit -m "docs(H-C7): update source_validation.py docstring — clarify xg_chain gate scope and pipeline-root location"
```

---

## Final Verification

- [ ] **Run the complete test suite**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all tests pass. Count of passing tests should be ≥ 208 (original Track H baseline) plus new tests added in this plan (~30 new tests).

- [ ] **Verify SOURCE_COLUMN_MAP is importable from package root**

```bash
python -c "from src.pipeline.datasources import SOURCE_COLUMN_MAP; print(list(SOURCE_COLUMN_MAP.keys()))"
```

Expected: `['fpl_post_gw', 'fpl_pre_gw', 'understat', 'espn', 'fpl_news', 'premierinjuries', 'ffs', 'reddit']`

- [ ] **Verify understat module uses soccerdata, not understatapi**

```bash
grep -r "understatapi" src/pipeline/datasources/
```

Expected: no output.

- [ ] **Verify soccerdata_client.py is gone**

```bash
ls src/pipeline/datasources/
```

Expected: no `soccerdata_client.py` in listing.

- [ ] **Final commit summary**

```bash
git log --oneline -8
```

Expected to see 7 commits: H-C1 through H-C7.

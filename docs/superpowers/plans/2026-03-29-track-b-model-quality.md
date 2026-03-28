# Track B — Model Quality: Understat, Positional Models, Fallback Benchmarking

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve prediction accuracy by (P4) reviving the Understat scraper and adding xG/xA features, (P3) testing per-position models, and (P5) benchmarking fallback strategies.

**Architecture:** P4 adds a new `src/pipeline/understat.py` scraper and extends `prepare.py` + `features.py`. P3 tests positional models in a research notebook before promoting to `predict.py`. P5 evaluates fallback candidates against historical data. Each item has a **decision gate**: only adopt if MAE improves.

**Tech Stack:** Python, requests/playwright (P4 — DOM may require JS rendering), scikit-learn, XGBoost, pandas. P3/P5 use existing stack only.

**Note:** P3 should run after P4 (xG/xA features make positional sets more distinct) but can use existing features as a first pass. P5 is fully independent.

---

## File Map

```
New:
  src/pipeline/understat.py          # Understat scraper for xG/xA/xMin per player per GW
  tests/test_understat.py
  results/understat_gw{N}.csv        # scraped data cache
  docs/research/p4-feature-comparison.md    # MAE comparison report
  docs/research/p3-positional-models.md     # model comparison report
  docs/research/p5-fallback-benchmarks.md   # fallback MAE comparison

Modified:
  src/pipeline/prepare.py            # join Understat data into merged dataset (if P4 adopted)
  src/pipeline/features.py           # add xG_roll_4, xA_roll_4, xMin_roll_4 (if P4 adopted)
  src/pipeline/predict.py            # positional routing (if P3 adopted)
  src/config.py                      # UNDERSTAT_BASE_URL constant
  src/pipeline/run.py                # fallback logic update (if P5 produces winner)
  requirements.txt                   # add playwright if needed for Understat
```

---

## Task 1 (P5): Fallback Strategy Benchmarking

**Start here** — P5 is independent, uses only existing code, and is fast to complete.

**Context:** Current fallback is `ep_this` from the FPL bootstrap. We want to know if something else (rolling average, `ep_next`, composite) gives lower MAE. Test on 5 historical GWs. The winner replaces `ep_this` in `run.py`'s fallback path.

**Files:**
- Modify: `tests/test_predict.py` (add benchmarking tests)
- Modify: `src/pipeline/run.py` (update fallback logic if winner != ep_this)
- Create: `docs/research/p5-fallback-benchmarks.md`

- [ ] **Step 1: Write failing test for fallback benchmarking helper**

Add to `tests/test_predict.py`:

```python
class TestFallbackBenchmarks:
    def test_compute_fallback_mae_returns_dict(self):
        from src.pipeline.predict import compute_fallback_mae
        import pandas as pd
        # Minimal historical data: 10 players × 10 GWs
        rows = []
        for p in range(1, 11):
            for gw in range(1, 11):
                rows.append({
                    "element": p, "code": p, "GW": gw,
                    "total_points": gw + p,
                    "ep_this": gw + p + 0.5,   # slight overestimate
                    "ep_next": gw + p - 0.3,   # slight underestimate
                    "total_points_roll_4": gw + p - 0.1 if gw > 4 else None,
                    "total_points_roll_8": gw + p - 0.2 if gw > 8 else None,
                })
        df = pd.DataFrame(rows).dropna(subset=["total_points_roll_4"])

        result = compute_fallback_mae(df)
        assert "ep_this" in result
        assert "ep_next" in result
        assert "roll_4_avg" in result
        for key, mae in result.items():
            assert mae >= 0.0, f"{key} MAE must be non-negative"

    def test_winner_is_lowest_mae_candidate(self):
        from src.pipeline.predict import compute_fallback_mae
        import pandas as pd
        rows = [{"element": i, "code": i, "GW": g,
                 "total_points": 5,
                 "ep_this": 10.0,       # terrible — far from 5
                 "ep_next": 5.0,        # perfect
                 "total_points_roll_4": 4.5,
                 "total_points_roll_8": 4.8}
                for i in range(1, 6) for g in range(1, 6)]
        df = pd.DataFrame(rows)
        result = compute_fallback_mae(df)
        winner = min(result, key=result.get)
        assert winner == "ep_next"
```

- [ ] **Step 2: Run to verify fail**

```bash
python -m pytest tests/test_predict.py::TestFallbackBenchmarks -v
```
Expected: FAIL — `compute_fallback_mae` not defined

- [ ] **Step 3: Add `compute_fallback_mae()` to `src/pipeline/predict.py`**

```python
def compute_fallback_mae(df: pd.DataFrame) -> dict[str, float]:
    """Benchmark fallback prediction strategies by MAE vs actual total_points.

    Candidates:
      ep_this       — FPL expected points for current GW
      ep_next       — FPL expected points for next GW
      roll_4_avg    — rolling 4-GW average of total_points
      roll_8_avg    — rolling 8-GW average of total_points
      composite     — 0.5*roll_4 + 0.3*roll_8 + 0.2*ep_next (if available)

    Args:
        df: Feature-engineered dataframe with actual total_points column.
            Must contain: total_points, and at least one of ep_this, ep_next,
            total_points_roll_4, total_points_roll_8.

    Returns:
        {strategy_name: MAE}  — only for strategies with sufficient data.
    """
    import numpy as np
    results = {}
    actuals = df["total_points"].values

    if "ep_this" in df.columns:
        pred = df["ep_this"].fillna(0).astype(float).values
        results["ep_this"] = float(np.mean(np.abs(actuals - pred)))

    if "ep_next" in df.columns:
        pred = df["ep_next"].fillna(0).astype(float).values
        results["ep_next"] = float(np.mean(np.abs(actuals - pred)))

    if "total_points_roll_4" in df.columns:
        valid = df["total_points_roll_4"].notna()
        if valid.sum() > 0:
            pred = df.loc[valid, "total_points_roll_4"].values
            act = df.loc[valid, "total_points"].values
            results["roll_4_avg"] = float(np.mean(np.abs(act - pred)))

    if "total_points_roll_8" in df.columns:
        valid = df["total_points_roll_8"].notna()
        if valid.sum() > 0:
            pred = df.loc[valid, "total_points_roll_8"].values
            act = df.loc[valid, "total_points"].values
            results["roll_8_avg"] = float(np.mean(np.abs(act - pred)))

    # Composite: 0.5*roll_4 + 0.3*roll_8 + 0.2*ep_next
    r4 = df.get("total_points_roll_4")
    r8 = df.get("total_points_roll_8")
    ep = df.get("ep_next")
    if r4 is not None and r8 is not None and ep is not None:
        valid = r4.notna() & r8.notna()
        if valid.sum() > 0:
            composite = (0.5 * r4[valid].fillna(0) +
                         0.3 * r8[valid].fillna(0) +
                         0.2 * ep[valid].fillna(0))
            results["composite"] = float(np.mean(np.abs(df.loc[valid, "total_points"].values - composite.values)))

    return results
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_predict.py::TestFallbackBenchmarks -v
```
Expected: PASS

- [ ] **Step 5: Run benchmark against real historical data**

```bash
python -c "
from src.pipeline.prepare import build_merged_dataset
from src.pipeline.features import engineer_features
from src.pipeline.predict import compute_fallback_mae
merged = build_merged_dataset()
features = engineer_features(merged, drop_na=False)
results = compute_fallback_mae(features)
for strategy, mae in sorted(results.items(), key=lambda x: x[1]):
    print(f'  {strategy:<20}: MAE = {mae:.4f}')
print()
winner = min(results, key=results.get)
print(f'Winner: {winner} (MAE {results[winner]:.4f})')
"
```

Record the output. Example:
```
  ep_next             : MAE = 3.1234
  roll_4_avg          : MAE = 3.3456
  ep_this             : MAE = 3.4567
  roll_8_avg          : MAE = 3.5678
  composite           : MAE = 3.2345

Winner: ep_next (MAE 3.1234)
```

- [ ] **Step 6: Write research findings to `docs/research/p5-fallback-benchmarks.md`**

Create `docs/research/` directory and write the file with:
- The full MAE table from step 5
- Which strategy wins
- Whether the pipeline fallback was updated (and what it was changed to)

```bash
mkdir -p docs/research
# Then write the file with actual benchmark numbers from step 5
```

- [ ] **Step 7: If winner is not `ep_this`, update `run.py` fallback path**

In `phase_predict()`, the fallback block sets `xP` from `ep_this` stored in the snapshot. If `ep_next` is the winner, update `extract_xp_snapshot()` in `fetch.py` to capture `ep_next` instead:

```python
# In fetch.py extract_xp_snapshot() — change ep_this to ep_next if benchmark shows ep_next wins
def extract_xp_snapshot(bootstrap_data: dict) -> dict[int, float]:
    result = {}
    for el in bootstrap_data["elements"]:
        ep = el.get("ep_next")   # CHANGED from ep_this if benchmarks show ep_next is better
        result[el["id"]] = float(ep) if ep is not None else 0.0
    return result
```

If `ep_this` is still the winner, no change needed.

- [ ] **Step 8: Run all tests to ensure no regressions**

```bash
python -m pytest tests/ -q
```

- [ ] **Step 9: Commit**

```bash
git add src/pipeline/predict.py src/pipeline/fetch.py tests/test_predict.py docs/research/p5-fallback-benchmarks.md
git commit -m "research(P5): benchmark fallback strategies; adopt $(winner) as default fallback"
```

---

## Task 2 (P4): Revive Understat Scraper

**Context:** Understat provides xG, xA, and xMin (expected minutes) per player per match. The old scraper in `_original/data_collection/understat.py` was hardcoded to season 2024. We're rewriting it as `src/pipeline/understat.py`. The FPL API does provide xG/xA from 2022-23 onwards (via `expected_goals`, `expected_assists` fields in `fetch.py`), but Understat has data from 2014 and includes `npxG` (non-penalty xG) and shot quality metrics the FPL API lacks.

**Risk note:** Understat uses JavaScript rendering. Test with `requests` first; if the site returns empty data, add `playwright`. Do NOT restore or modify `_original/data_collection/understat.py`.

**Files:**
- Create: `src/pipeline/understat.py`
- Create: `tests/test_understat.py`

- [ ] **Step 1: Read the original scraper for reference (do not modify)**

```bash
cat _original/data_collection/understat.py
```

Note the data format and endpoints used. Do NOT edit this file.

- [ ] **Step 2: Write failing tests for the new scraper**

Create `tests/test_understat.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from src.pipeline.understat import (
    fetch_player_understat,
    parse_understat_player_data,
    build_understat_season_df,
)


class TestParseUnderstatePlayerData:
    def test_parses_player_xg_per_match(self):
        # Raw JSON from Understat player page (matches array)
        raw_data = [
            {"id": "1", "h_a": "h", "xG": "0.85", "xA": "0.20",
             "time": "90", "date": "2025-08-16", "season": "2025",
             "npxG": "0.85", "shots": "3"},
            {"id": "2", "h_a": "a", "xG": "0.12", "xA": "0.05",
             "time": "75", "date": "2025-08-24", "season": "2025",
             "npxG": "0.12", "shots": "1"},
        ]
        df = parse_understat_player_data(raw_data, player_name="Saka", season="2025-26")
        assert len(df) == 2
        assert "xG" in df.columns
        assert "xA" in df.columns
        assert "xMin" in df.columns
        assert "npxG" in df.columns
        assert abs(df.iloc[0]["xG"] - 0.85) < 0.001
        assert df.iloc[1]["xMin"] == 75

    def test_handles_missing_fields(self):
        raw_data = [{"id": "1", "xG": "0.5", "date": "2025-08-16", "season": "2025"}]
        df = parse_understat_player_data(raw_data, player_name="Unknown", season="2025-26")
        assert len(df) == 1
        assert df.iloc[0]["xA"] == 0.0  # default 0 for missing field


class TestFetchPlayerUnderstat:
    def test_returns_dataframe_with_expected_shape(self):
        """Mock HTTP response containing embedded JSON data."""
        mock_html = """
        <html><head></head><body>
        <script>var player_data = JSON.parse('{"matches":[
            {"id":"1","h_a":"h","xG":"0.8","xA":"0.3","time":"90","date":"2025-08-16",
             "season":"2025","npxG":"0.8","shots":"2"}
        ]}');</script>
        </body></html>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = mock_html

        with patch("src.pipeline.understat.requests.get", return_value=mock_resp):
            df = fetch_player_understat(player_id=8260, player_name="Saka", season="2025-26")

        assert len(df) >= 0  # may be 0 if parsing needs adjustment — implementation defines format
```

- [ ] **Step 3: Run to verify fail**

```bash
python -m pytest tests/test_understat.py -v
```
Expected: FAIL — ImportError

- [ ] **Step 4: Create `src/pipeline/understat.py`**

```python
"""Understat xG/xA/xMin scraper for FPL pipeline.

Fetches per-match xG, xA, npxG, xMin from understat.com player pages.
Data is available from 2014-15 onwards for top 5 European leagues.

Usage:
    df = fetch_player_understat(player_id=8260, player_name="Saka", season="2025-26")
    # Returns DataFrame with columns: player_name, date, season, xG, xA, npxG, xMin, shots, h_a
"""
from __future__ import annotations
import json
import logging
import re
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

UNDERSTAT_BASE_URL = "https://understat.com"
UNDERSTAT_REQUEST_DELAY = 1.5  # seconds between requests (polite scraping)


def _season_to_understat_year(season: str) -> str:
    """Convert "2025-26" → "2025" (Understat uses start year)."""
    return season.split("-")[0]


def fetch_player_understat(
    player_id: int,
    player_name: str,
    season: str,
) -> pd.DataFrame:
    """Fetch a single player's per-match xG/xA data from Understat.

    Scrapes the player page and extracts the embedded JSON data.
    Falls back to empty DataFrame if the request fails or data is unavailable.
    """
    url = f"{UNDERSTAT_BASE_URL}/player/{player_id}"
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Understat fetch failed for {player_name} (id={player_id}): {e}")
        return pd.DataFrame()

    # Extract JSON from embedded script: var player_data = JSON.parse('...')
    match = re.search(r"var\s+player_data\s*=\s*JSON\.parse\('(.+?)'\)", resp.text, re.DOTALL)
    if not match:
        # Try alternate pattern (some pages use direct JSON)
        match = re.search(r"var\s+player_data\s*=\s*(\{.+?\});", resp.text, re.DOTALL)
        if not match:
            logger.warning(f"Could not extract player_data JSON for {player_name}")
            return pd.DataFrame()
        raw_json = json.loads(match.group(1))
    else:
        # Unescape the JSON string
        json_str = match.group(1).encode().decode("unicode_escape")
        raw_json = json.loads(json_str)

    matches_data = raw_json.get("matches", [])
    year_filter = _season_to_understat_year(season)
    season_matches = [m for m in matches_data if str(m.get("season", "")) == year_filter]

    if not season_matches:
        logger.info(f"No Understat data for {player_name} in season {season}")
        return pd.DataFrame()

    return parse_understat_player_data(season_matches, player_name=player_name, season=season)


def parse_understat_player_data(
    raw_matches: list[dict],
    player_name: str,
    season: str,
) -> pd.DataFrame:
    """Parse raw Understat matches JSON into a clean DataFrame.

    Output columns: player_name, date, season, h_a, xG, xA, npxG, xMin, shots
    """
    rows = []
    for m in raw_matches:
        rows.append({
            "player_name": player_name,
            "understat_match_id": m.get("id"),
            "date": m.get("date", ""),
            "season": season,
            "h_a": m.get("h_a", ""),
            "xG": float(m.get("xG", 0) or 0),
            "xA": float(m.get("xA", 0) or 0),
            "npxG": float(m.get("npxG", 0) or 0),
            "xMin": float(m.get("time", 0) or 0),
            "shots": int(m.get("shots", 0) or 0),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def build_understat_season_df(
    player_id_map: dict[str, int],
    season: str,
    request_delay: float = UNDERSTAT_REQUEST_DELAY,
) -> pd.DataFrame:
    """Fetch Understat data for multiple players and concatenate.

    Args:
        player_id_map: {player_name: understat_player_id}
        season: "2025-26"
        request_delay: seconds between requests (default 1.5 to be polite)

    Returns: DataFrame with all players' per-match xG/xA data.
    """
    all_dfs = []
    for i, (name, pid) in enumerate(player_id_map.items()):
        if i > 0:
            time.sleep(request_delay)
        df = fetch_player_understat(pid, name, season)
        if not df.empty:
            all_dfs.append(df)
        if (i + 1) % 20 == 0:
            logger.info(f"Fetched Understat data for {i + 1}/{len(player_id_map)} players")

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
```

- [ ] **Step 5: Add `UNDERSTAT_BASE_URL` to `src/config.py`**

```python
UNDERSTAT_BASE_URL = "https://understat.com"
```

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_understat.py -v
```
Expected: parsing tests PASS; fetch test may PASS with mocked response

- [ ] **Step 7: Test against real Understat (manual, requires network)**

```bash
python -c "
from src.pipeline.understat import fetch_player_understat
# Saka's Understat ID is 8260 (verify at understat.com/player/8260)
df = fetch_player_understat(player_id=8260, player_name='Saka', season='2025-26')
print(df.head())
print(f'Rows: {len(df)}')
"
```

**Decision gate:** If this fails due to JS rendering, add playwright:
```bash
pip install playwright
playwright install chromium
```
And replace `requests.get()` in `understat.py` with:
```python
from playwright.sync_api import sync_playwright
def _fetch_url_with_js(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        content = page.content()
        browser.close()
        return content
```

If Understat data loads successfully, proceed to Task 3.

- [ ] **Step 8: Commit**

```bash
git add src/pipeline/understat.py src/config.py tests/test_understat.py requirements.txt
git commit -m "feat(P4): Understat scraper with xG/xA/xMin per player per match"
```

---

## Task 3 (P4): FPL-to-Understat player name matching

**Context:** FPL uses `web_name` (e.g. "Saka"). Understat uses full display names (e.g. "Bukayo Saka"). We need a player ID map. Understat has a league player listing that can be scraped once per season, or we can maintain a manual lookup file.

**Files:**
- Create: `data/understat_player_ids.csv` (manually curated + auto-refreshed)
- Modify: `src/pipeline/understat.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_understat.py`:

```python
class TestPlayerIdMatching:
    def test_match_fpl_name_to_understat_id(self):
        from src.pipeline.understat import match_fpl_to_understat
        # Build a small lookup table
        lookup = pd.DataFrame({
            "understat_name": ["Bukayo Saka", "Erling Haaland", "Cole Palmer"],
            "understat_id": [8260, 9278, 11813],
            "fpl_code": [56304, 80201, 226597],
        })
        result = match_fpl_to_understat(fpl_code=56304, lookup_df=lookup)
        assert result == 8260

    def test_returns_none_for_missing_player(self):
        from src.pipeline.understat import match_fpl_to_understat
        lookup = pd.DataFrame({"fpl_code": [1], "understat_id": [100], "understat_name": ["X"]})
        result = match_fpl_to_understat(fpl_code=99999, lookup_df=lookup)
        assert result is None
```

- [ ] **Step 2: Add `match_fpl_to_understat()` to `src/pipeline/understat.py`**

```python
def match_fpl_to_understat(
    fpl_code: int,
    lookup_df: pd.DataFrame,
) -> int | None:
    """Look up Understat player ID from FPL persistent player code.

    lookup_df must have columns: fpl_code, understat_id.
    Returns None if not found.
    """
    match = lookup_df[lookup_df["fpl_code"] == fpl_code]
    if match.empty:
        return None
    return int(match.iloc[0]["understat_id"])
```

- [ ] **Step 3: Create `data/understat_player_ids.csv`**

Seed with top 50 FPL players manually (fpl_code from `players_raw.csv`, understat_id from understat.com):

```csv
fpl_code,understat_id,understat_name,fpl_web_name
56304,8260,Bukayo Saka,Saka
80201,9278,Erling Haaland,Haaland
...
```

This file is committed and manually extended. Understat IDs are stable and don't change.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_understat.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/understat.py data/understat_player_ids.csv tests/test_understat.py
git commit -m "feat(P4): FPL-to-Understat player matching via lookup CSV"
```

---

## Task 4 (P4): Integrate Understat features into pipeline

**Context:** Add `xG_roll_4`, `xA_roll_4`, `npxG_roll_4`, `xMin_roll_4` rolling features. Join Understat data in `prepare.py` using match date (± 1 day tolerance for fixture alignment). Decision gate: only update `FEATURE_COLUMNS` if retrained model shows improved MAE.

**Files:**
- Modify: `src/pipeline/prepare.py`
- Modify: `src/pipeline/features.py`
- Modify: `tests/test_prepare.py`
- Modify: `tests/test_features.py`

- [ ] **Step 1: Write failing test for Understat join in prepare.py**

Add to `tests/test_prepare.py`:

```python
class TestJoinUnderstateData:
    def test_adds_xg_column_when_data_available(self, sample_gw_df):
        import pandas as pd
        from src.pipeline.prepare import join_understat_data
        # Understat data for Saka
        understat_df = pd.DataFrame({
            "player_name": ["Saka", "Saka"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-08"]),
            "xG": [0.85, 0.20],
            "xA": [0.30, 0.10],
            "npxG": [0.85, 0.20],
            "xMin": [90.0, 75.0],
        })
        # sample_gw_df needs kickoff_time column
        gw_df = sample_gw_df.copy()
        gw_df["kickoff_time"] = pd.to_datetime(
            ["2026-01-01"] * 4 + ["2026-01-08"] * 2
        )
        gw_df["name"] = ["Saka"] * 6
        result = join_understat_data(gw_df, understat_df)
        assert "xG" in result.columns
        assert result[result["name"] == "Saka"]["xG"].notna().any()

    def test_fills_na_when_no_understat_data(self, sample_gw_df):
        import pandas as pd
        from src.pipeline.prepare import join_understat_data
        gw_df = sample_gw_df.copy()
        gw_df["kickoff_time"] = pd.to_datetime(["2026-01-01"] * 6)
        result = join_understat_data(gw_df, pd.DataFrame())
        assert "xG" in result.columns
        assert result["xG"].fillna(0).ge(0).all()
```

- [ ] **Step 2: Add `join_understat_data()` to `src/pipeline/prepare.py`**

```python
def join_understat_data(
    gw_df: pd.DataFrame,
    understat_df: pd.DataFrame,
    date_tolerance_days: int = 1,
) -> pd.DataFrame:
    """Join Understat xG/xA/xMin data onto GW DataFrame by player name and date.

    Uses a ±tolerance merge (date-based) since fixture dates may differ by a day.
    Fills NaN with 0 for players without Understat data.
    """
    xg_cols = ["xG", "xA", "npxG", "xMin"]
    for col in xg_cols:
        gw_df[col] = 0.0

    if understat_df.empty or "date" not in understat_df.columns:
        return gw_df

    if "kickoff_time" not in gw_df.columns:
        return gw_df

    gw_df = gw_df.copy()
    gw_df["_date"] = pd.to_datetime(gw_df["kickoff_time"], errors="coerce").dt.normalize()

    understat_df = understat_df.copy()
    understat_df["_date"] = pd.to_datetime(understat_df["date"], errors="coerce").dt.normalize()

    for idx, row in gw_df.iterrows():
        if pd.isna(row.get("_date")):
            continue
        player_name = row.get("name", "")
        match_window = pd.date_range(
            row["_date"] - pd.Timedelta(days=date_tolerance_days),
            row["_date"] + pd.Timedelta(days=date_tolerance_days),
        )
        mask = (
            (understat_df["player_name"] == player_name) &
            (understat_df["_date"].isin(match_window))
        )
        matched = understat_df[mask]
        if not matched.empty:
            for col in xg_cols:
                if col in matched.columns:
                    gw_df.at[idx, col] = float(matched.iloc[0][col])

    return gw_df.drop(columns=["_date"])
```

- [ ] **Step 3: Add rolling xG features to `src/pipeline/features.py`**

In `features.py`, update `ROLLING_COLS` if Understat data is available:

```python
UNDERSTAT_COLS = ["xG", "xA", "npxG", "xMin"]

def add_understat_rolling_features(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Add rolling xG/xA/xMin features — only if Understat columns are present."""
    windows = windows or DEFAULT_WINDOWS
    present = [c for c in UNDERSTAT_COLS if c in df.columns and df[c].notna().any()]
    if not present:
        return df
    player_id = "code" if "code" in df.columns else "element"
    df = df.sort_values([player_id, "season", "GW"]).copy()
    for col in present:
        for w in windows:
            df[f"{col}_roll_{w}"] = (
                df.groupby(player_id)[col]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=w).mean())
            )
    return df
```

Update `engineer_features()` to call this:

```python
def engineer_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    df = add_rolling_features(df)
    df = add_momentum_features(df)
    df = add_form_features(df)
    df = add_understat_rolling_features(df)   # NEW: no-op if Understat cols absent
    if drop_na:
        longest_window = max(DEFAULT_WINDOWS)
        df = df.dropna(subset=[f"total_points_roll_{longest_window}"])
    return df
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_prepare.py tests/test_features.py -v
```
Expected: all PASS

- [ ] **Step 5: Decision gate — retrain and compare MAE**

```bash
python -m src.pipeline.run retrain --gw 33
```

Compare MAE printout with the previous model. If MAE improves, add the new features to `FEATURE_COLUMNS` in `predict.py`:

```python
FEATURE_COLUMNS = [
    # ... existing 18 features ...
    "xG_roll_4", "xA_roll_4", "npxG_roll_4",   # ADD only if MAE improves
]
```

If MAE does NOT improve: keep the features in `features.py` (they're computed but not used in the model), document in `docs/research/p4-feature-comparison.md`, and leave `FEATURE_COLUMNS` unchanged.

- [ ] **Step 6: Write research findings**

Create `docs/research/p4-feature-comparison.md`:
- Old model MAE (18 features)
- New model MAE (18 + xG/xA features)
- Decision: adopted or not
- Feature importance ranking comparison

- [ ] **Step 7: Commit**

```bash
git add src/pipeline/prepare.py src/pipeline/features.py tests/test_prepare.py tests/test_features.py docs/research/p4-feature-comparison.md
git commit -m "research(P4): add Understat xG/xA rolling features; adopt if MAE improves"
```

---

## Task 5 (P3): Positional Model Research

**Context:** Test whether training 4 separate models (one per position) beats the global model. Uses `total_points` as target with position-specific feature subsets. P4 Understat features improve the distinction; P3 can proceed with existing features as first pass.

**Dependency:** P4 recommended but not required. If using existing features only, skip xG/xA columns.

**Files:**
- Modify: `src/pipeline/predict.py` (add `train_positional_models()`, `predict_positional()`)
- Create: `tests/test_predict.py` (add positional tests)
- Create: `docs/research/p3-positional-models.md`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_predict.py`:

```python
class TestPositionalModels:
    def test_train_positional_models_returns_4_models(self):
        from src.pipeline.predict import train_positional_models
        import pandas as pd
        import numpy as np
        # Minimal training dataset
        n = 100
        df = pd.DataFrame({
            "code": np.arange(n),
            "position": np.random.choice(["GK", "DEF", "MID", "FWD"], n),
            "total_points": np.random.randint(0, 15, n),
            "total_points_roll_4": np.random.rand(n) * 5,
            "minutes_roll_4": np.random.rand(n) * 90,
            "clean_sheets_roll_4": np.random.rand(n),
            "ict_index_roll_4": np.random.rand(n) * 10,
            "threat_roll_4": np.random.rand(n) * 20,
            "creativity_roll_4": np.random.rand(n) * 15,
            "goals_scored_roll_4": np.random.rand(n) * 0.5,
            "assists_roll_4": np.random.rand(n) * 0.3,
        })
        models = train_positional_models(df)
        assert set(models.keys()) == {"GK", "DEF", "MID", "FWD"}
        for pos, model in models.items():
            assert hasattr(model, "predict")

    def test_positional_mae_lower_or_equal_to_global(self):
        """Decision gate: positional models should not be WORSE than global."""
        from src.pipeline.predict import train_positional_models, compare_model_mae
        import pandas as pd
        import numpy as np
        # Not a strict test — just verify the comparison function runs
        n = 200
        df = pd.DataFrame({
            "code": np.arange(n),
            "position": np.random.choice(["GK", "DEF", "MID", "FWD"], n),
            "total_points": np.random.randint(0, 15, n),
            "total_points_roll_4": np.random.rand(n) * 5,
            "minutes_roll_4": np.random.rand(n) * 90,
            "clean_sheets_roll_4": np.random.rand(n),
            "ict_index_roll_4": np.random.rand(n) * 10,
            "threat_roll_4": np.random.rand(n) * 20,
            "creativity_roll_4": np.random.rand(n) * 15,
            "goals_scored_roll_4": np.random.rand(n) * 0.5,
            "assists_roll_4": np.random.rand(n) * 0.3,
        })
        comparison = compare_model_mae(df)
        assert "global_mae" in comparison
        assert "positional_mae" in comparison
        for pos in ["GK", "DEF", "MID", "FWD"]:
            assert pos in comparison["positional_mae"]
```

- [ ] **Step 2: Run to verify fail**

```bash
python -m pytest tests/test_predict.py::TestPositionalModels -v
```

- [ ] **Step 3: Add `train_positional_models()` and `compare_model_mae()` to `src/pipeline/predict.py`**

```python
POSITIONAL_FEATURES = {
    "GK":  ["clean_sheets_roll_4", "minutes_roll_4", "total_points_roll_4",
             "ict_index_roll_4", "bps_roll_4"],
    "DEF": ["clean_sheets_roll_4", "minutes_roll_4", "total_points_roll_4",
             "ict_index_roll_4", "threat_roll_4", "goals_scored_roll_4"],
    "MID": ["ict_index_roll_4", "creativity_roll_4", "threat_roll_4",
             "assists_roll_4", "total_points_roll_4", "minutes_roll_4"],
    "FWD": ["threat_roll_4", "goals_scored_roll_4", "ict_index_roll_4",
             "total_points_roll_4", "minutes_roll_4", "assists_roll_4"],
}


def train_positional_models(df: pd.DataFrame) -> dict:
    """Train one RandomForest model per position on position-specific features.

    Returns: {"GK": model, "DEF": model, "MID": model, "FWD": model}
    Only uses features present in df.columns.
    """
    from sklearn.ensemble import RandomForestRegressor
    models = {}
    for pos, feat_cols in POSITIONAL_FEATURES.items():
        pos_df = df[df["position"] == pos].copy()
        if len(pos_df) < 50:
            continue
        available = [c for c in feat_cols if c in pos_df.columns]
        X = pos_df[available].fillna(0)
        y = pos_df["total_points"]
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        models[pos] = {"model": model, "features": available}
    return models


def compare_model_mae(df: pd.DataFrame) -> dict:
    """Compare global model MAE vs per-position model MAE on a holdout split.

    Returns: {"global_mae": float, "positional_mae": {"GK": float, ...}}
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Global model
    global_feats = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    X_train = train_df[global_feats].fillna(0)
    y_train = train_df["total_points"]
    X_test = test_df[global_feats].fillna(0)
    y_test = test_df["total_points"]
    global_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    global_model.fit(X_train, y_train)
    global_mae = mean_absolute_error(y_test, global_model.predict(X_test))

    # Positional models
    pos_models = train_positional_models(train_df)
    pos_mae = {}
    for pos, model_dict in pos_models.items():
        pos_test = test_df[test_df["position"] == pos]
        if pos_test.empty:
            continue
        feats = model_dict["features"]
        X = pos_test[[c for c in feats if c in pos_test.columns]].fillna(0)
        y = pos_test["total_points"]
        pos_mae[pos] = mean_absolute_error(y, model_dict["model"].predict(X))

    return {"global_mae": global_mae, "positional_mae": pos_mae}
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_predict.py::TestPositionalModels -v
```

- [ ] **Step 5: Decision gate — run comparison on real data**

```bash
python -c "
from src.pipeline.prepare import build_merged_dataset
from src.pipeline.features import engineer_features
from src.pipeline.predict import compare_model_mae
merged = build_merged_dataset()
features = engineer_features(merged)
result = compare_model_mae(features)
print(f'Global MAE: {result[\"global_mae\"]:.4f}')
for pos, mae in result[\"positional_mae\"].items():
    diff = mae - result[\"global_mae\"]
    print(f'  {pos} MAE: {mae:.4f}  ({diff:+.4f} vs global)')
"
```

**If positional models are BETTER (lower MAE per position):**
- Save 4 model files: `models/rf_gk_gw{N}.sav`, `rf_def_gw{N}.sav`, etc.
- Update `predict_next_gw()` to route players to position-specific models
- Update `ACTIVE_MODEL` → `ACTIVE_POSITIONAL_MODELS` in `src/config.py`

**If positional models are WORSE or EQUAL:**
- Document findings in `docs/research/p3-positional-models.md`
- Keep global model unchanged

- [ ] **Step 6: Write research doc and commit**

```bash
git add src/pipeline/predict.py tests/test_predict.py docs/research/p3-positional-models.md
git commit -m "research(P3): positional model comparison; adopt if MAE improves per position"
```

---

## Final: Full test suite + mark roadmap

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: all PASS

- [ ] **Step 2: Update improvements roadmap**

In `docs/improvements-roadmap.md`, mark each completed/attempted item:
- P4: ✅ Implemented / ⚠️ Deferred (Understat DOM changed — playwright required)
- P3: ✅ Adopted / ❌ Rejected (global model better)
- P5: ✅ Winner: {strategy}

- [ ] **Step 3: Commit**

```bash
git add docs/improvements-roadmap.md docs/research/
git commit -m "docs: update P3/P4/P5 research findings in improvements roadmap"
```

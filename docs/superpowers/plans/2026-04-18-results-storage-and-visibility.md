# Results Storage & Performance Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganise `results/` into a season/GW folder structure, move snapshots to `data/`, add per-GW and cumulative rank-comparison PNGs, and track actual vs recommended transfers with point impact.

**Architecture:** Config helper functions (`gw_dir`, `snapshot_dir`) centralise all path logic; every pipeline phase is updated to use them. New `generate_reports.py` script reads `accuracy_log.csv` and `actual_transfers.csv` to produce two matplotlib PNGs. One-off migration and backfill scripts handle the existing GW30–33 data.

**Tech Stack:** Python, pandas, matplotlib, existing `fetch.py` retry wrapper, pytest.

**Spec:** `docs/superpowers/specs/2026-04-18-results-storage-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/config.py` | Modify | Add `gw_dir()`, `snapshot_dir()`; move `SNAPSHOTS_DIR` to `data/`; remove `SIGNAL_UNRESOLVED_CSV` |
| `src/pipeline/datasources/signals.py` | Modify | Replace `log_unresolved_name` body with `logging.warning()` |
| `src/pipeline/analysis.py` | Modify | Add `season` param to `append_accuracy_log`; add `build_actual_squad_csv()` |
| `src/pipeline/run.py` | Modify | Add `_build_squad_csv()`, `_fetch_actual_transfers()`; update all output paths; update phase_post_gw |
| `scripts/generate_reports.py` | Create | Two-panel rank PNGs from `accuracy_log.csv` + `actual_transfers.csv` |
| `scripts/migrate_results.py` | Create then delete | One-off: reorganise existing files, move snapshots, merge squad CSVs |
| `scripts/backfill_actuals.py` | Create then delete | One-off: backfill GW31–33 `actual_squad.csv` + `actual_transfers.csv` |
| `.github/workflows/daily_bootstrap.yml` | Modify | Update snapshot paths; add `generate_reports.py` call after post-gw |
| `tests/test_config.py` | Modify | Tests for `gw_dir()` and `snapshot_dir()` |
| `tests/test_analysis.py` | Modify | Tests for `season` in `append_accuracy_log` and `build_actual_squad_csv` |
| `tests/test_run_squad_csv.py` | Create | Tests for `_build_squad_csv()` and `_fetch_actual_transfers()` |
| `tests/test_generate_reports.py` | Create | Tests for chart data preparation logic |

---

## Task 1: Config helpers + signals.py cleanup

**Files:**
- Modify: `src/config.py`
- Modify: `src/pipeline/datasources/signals.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for gw_dir and snapshot_dir**

```python
# tests/test_config.py — add to existing file
from src.config import gw_dir, snapshot_dir, SNAPSHOTS_DIR, DATA_DIR

def test_gw_dir_returns_correct_path():
    p = gw_dir("2025-26", 31)
    assert str(p).replace("\\", "/").endswith("results/2025-26/gw31")

def test_snapshot_dir_returns_correct_path():
    p = snapshot_dir("2025-26", 31)
    assert str(p).replace("\\", "/").endswith("data/snapshots/2025-26/gw31")

def test_snapshots_dir_is_under_data():
    assert str(SNAPSHOTS_DIR).replace("\\", "/").endswith("data/snapshots")

def test_signal_unresolved_csv_removed():
    import src.config as cfg
    assert not hasattr(cfg, "SIGNAL_UNRESOLVED_CSV")
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/test_config.py::test_gw_dir_returns_correct_path tests/test_config.py::test_snapshot_dir_returns_correct_path tests/test_config.py::test_snapshots_dir_is_under_data tests/test_config.py::test_signal_unresolved_csv_removed -v
```

Expected: FAIL (AttributeError / AssertionError)

- [ ] **Step 3: Update config.py**

In `src/config.py`, make these changes:

```python
# Change line 8 from:
SNAPSHOTS_DIR = RESULTS_DIR / "snapshots"
# To:
SNAPSHOTS_DIR = DATA_DIR / "snapshots"

# Add after RESULTS_DIR definition:
CURRENT_SEASON = "2025-26"   # already exists — confirm it's here, add if not

def gw_dir(season: str, gw: int) -> Path:
    return RESULTS_DIR / season / f"gw{gw}"

def snapshot_dir(season: str, gw: int) -> Path:
    return SNAPSHOTS_DIR / season / f"gw{gw}"
```

Remove the line:
```python
SIGNAL_UNRESOLVED_CSV = RESULTS_DIR / "signal_unresolved.csv"
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_config.py::test_gw_dir_returns_correct_path tests/test_config.py::test_snapshot_dir_returns_correct_path tests/test_config.py::test_snapshots_dir_is_under_data tests/test_config.py::test_signal_unresolved_csv_removed -v
```

Expected: PASS

- [ ] **Step 5: Fix signals.py**

In `src/pipeline/datasources/signals.py`, replace `log_unresolved_name` body (lines ~56–75):

```python
def log_unresolved_name(
    name: str,
    source: str,
    raw_text: str,
    csv_path=None,          # kept for backwards-compat; ignored
    timestamp: str = "",
) -> None:
    import logging
    logging.getLogger(__name__).warning(
        "[%s] Unresolved player: %r — %s", source, name, raw_text[:80]
    )
```

- [ ] **Step 6: Run full test suite to catch any import breakage**

```bash
python -m pytest tests/ -q --tb=short 2>&1 | tail -20
```

Expected: same pass count as before (no new failures from removed `SIGNAL_UNRESOLVED_CSV`)

- [ ] **Step 7: Commit**

```bash
git add src/config.py src/pipeline/datasources/signals.py tests/test_config.py
git commit -m "feat: add gw_dir/snapshot_dir helpers, move SNAPSHOTS_DIR to data/, remove SIGNAL_UNRESOLVED_CSV"
```

---

## Task 2: analysis.py — season in accuracy_log + build_actual_squad_csv

**Files:**
- Modify: `src/pipeline/analysis.py`
- Modify: `tests/test_analysis.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_analysis.py — add to existing file
from src.pipeline.analysis import append_accuracy_log, build_actual_squad_csv

def test_append_accuracy_log_writes_season_column(tmp_path):
    log = tmp_path / "accuracy_log.csv"
    append_accuracy_log(log, gw=31, season="2025-26",
                        your_pts=44, your_xp=44.3,
                        recommended_pts=8, recommended_xp=38.8)
    import pandas as pd
    df = pd.read_csv(log)
    assert "season" in df.columns
    assert df.iloc[0]["season"] == "2025-26"

def test_append_accuracy_log_season_defaults_to_current_season(tmp_path):
    from src.config import CURRENT_SEASON
    log = tmp_path / "accuracy_log.csv"
    append_accuracy_log(log, gw=31, your_pts=44, your_xp=44.3,
                        recommended_pts=8, recommended_xp=38.8)
    import pandas as pd
    df = pd.read_csv(log)
    assert df.iloc[0]["season"] == CURRENT_SEASON

def test_build_actual_squad_csv_columns():
    entry_picks = {
        "picks": [
            {"element": 1, "position": 1, "multiplier": 2, "is_captain": True, "is_vice_captain": False},
            {"element": 2, "position": 12, "multiplier": 0, "is_captain": False, "is_vice_captain": True},
        ]
    }
    bootstrap = {
        "elements": [
            {"id": 1, "web_name": "Salah", "element_type": 3, "team": 14, "now_cost": 130},
            {"id": 2, "web_name": "Saka",  "element_type": 3, "team": 1,  "now_cost": 100},
        ],
        "teams": [{"id": 1, "name": "Arsenal"}, {"id": 14, "name": "Liverpool"}],
        "element_types": [
            {"id": 3, "singular_name_short": "MID"},
        ],
    }
    actual_pts = {1: 20, 2: 6}
    df = build_actual_squad_csv(entry_picks, bootstrap, actual_pts)
    assert list(df.columns) == [
        "element", "name", "position", "team", "actual_pts",
        "is_starter", "bench_order", "is_captain", "is_vice_captain", "now_cost"
    ]
    salah = df[df["element"] == 1].iloc[0]
    assert salah["actual_pts"] == 20
    assert salah["is_captain"] is True
    assert salah["is_starter"] is True
    saka = df[df["element"] == 2].iloc[0]
    assert saka["is_starter"] is False
    assert saka["bench_order"] == 1
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/test_analysis.py::test_append_accuracy_log_writes_season_column tests/test_analysis.py::test_append_accuracy_log_season_defaults_to_current_season tests/test_analysis.py::test_build_actual_squad_csv_columns -v
```

Expected: FAIL

- [ ] **Step 3: Add season param to append_accuracy_log**

In `src/pipeline/analysis.py`, update `append_accuracy_log` signature and row dict:

```python
def append_accuracy_log(
    path: Path,
    gw: int,
    your_pts: int | None,
    your_xp: float | None,
    recommended_pts: int | None,
    recommended_xp: float | None,
    season: str | None = None,   # ← ADD THIS
    # ... existing params unchanged ...
) -> None:
    from src.config import CURRENT_SEASON
    if season is None:
        season = CURRENT_SEASON
    # ...
    row = {
        "gw": gw,
        "season": season,          # ← ADD AS SECOND KEY
        "your_pts": your_pts,
        # ... rest unchanged ...
    }
```

- [ ] **Step 4: Add build_actual_squad_csv to analysis.py**

```python
def build_actual_squad_csv(
    entry_picks: dict,
    bootstrap: dict,
    actual_pts_by_element: dict,
) -> pd.DataFrame:
    """Assemble actual_squad.csv schema from FPL picks API response."""
    el_map = {e["id"]: e for e in bootstrap.get("elements", [])}
    team_map = {t["id"]: t["name"] for t in bootstrap.get("teams", [])}
    pos_map = {pt["id"]: pt["singular_name_short"]
               for pt in bootstrap.get("element_types", [])}

    picks = entry_picks.get("picks", [])
    # positions 1-11 = starters, 12-15 = bench
    starters = {p["element"] for p in picks if p["position"] <= 11}
    bench_picks = sorted(
        [p for p in picks if p["position"] > 11],
        key=lambda p: p["position"]
    )
    bench_order = {p["element"]: i + 1 for i, p in enumerate(bench_picks)}

    rows = []
    for pick in picks:
        el_id = pick["element"]
        el = el_map.get(el_id, {})
        rows.append({
            "element": el_id,
            "name": el.get("web_name", str(el_id)),
            "position": pos_map.get(el.get("element_type"), "?"),
            "team": team_map.get(el.get("team"), "?"),
            "actual_pts": actual_pts_by_element.get(el_id),
            "is_starter": el_id in starters,
            "bench_order": bench_order.get(el_id),
            "is_captain": pick.get("is_captain", False),
            "is_vice_captain": pick.get("is_vice_captain", False),
            "now_cost": el.get("now_cost", 0) / 10,
        })
    return pd.DataFrame(rows, columns=[
        "element", "name", "position", "team", "actual_pts",
        "is_starter", "bench_order", "is_captain", "is_vice_captain", "now_cost"
    ])
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python -m pytest tests/test_analysis.py::test_append_accuracy_log_writes_season_column tests/test_analysis.py::test_append_accuracy_log_season_defaults_to_current_season tests/test_analysis.py::test_build_actual_squad_csv_columns -v
```

Expected: PASS

- [ ] **Step 6: Run full suite**

```bash
python -m pytest tests/ -q --tb=short 2>&1 | tail -10
```

Expected: no regressions in existing accuracy_log tests.

- [ ] **Step 7: Commit**

```bash
git add src/pipeline/analysis.py tests/test_analysis.py
git commit -m "feat: add season to accuracy_log, add build_actual_squad_csv"
```

---

## Task 3: run.py — _build_squad_csv helper + phase_predict/recommend paths

**Files:**
- Modify: `src/pipeline/run.py`
- Create: `tests/test_run_squad_csv.py`

- [ ] **Step 1: Write failing test for _build_squad_csv**

```python
# tests/test_run_squad_csv.py
import pandas as pd

def _make_optimize_result():
    xi = pd.DataFrame([
        {"element": 1, "name": "Salah",   "position": "MID", "team": "Liverpool", "xP": 10.0, "now_cost": 13.0},
        {"element": 2, "name": "Saka",    "position": "MID", "team": "Arsenal",   "xP": 8.0,  "now_cost": 10.0},
    ])
    bench = pd.DataFrame([
        {"element": 3, "name": "Flekken", "position": "GK",  "team": "Brentford", "xP": 3.0,  "now_cost": 4.5},
        {"element": 4, "name": "Mykolenko","position": "DEF","team": "Everton",   "xP": 2.0,  "now_cost": 4.0},
    ])
    return {
        "xi": xi, "bench": bench,
        "captain": xi.iloc[0], "vice_captain": xi.iloc[1],
        "squad": pd.concat([xi, bench]), "total_xp": 21.0,
    }

def test_build_squad_csv_columns():
    from src.pipeline.run import _build_squad_csv
    df = _build_squad_csv(_make_optimize_result())
    assert list(df.columns) == [
        "element", "name", "position", "team", "xP",
        "is_starter", "bench_order", "is_captain", "is_vice_captain", "now_cost"
    ]

def test_build_squad_csv_starters_have_null_bench_order():
    from src.pipeline.run import _build_squad_csv
    df = _build_squad_csv(_make_optimize_result())
    starters = df[df["is_starter"]]
    assert starters["bench_order"].isna().all()

def test_build_squad_csv_bench_ranked_by_xp_desc():
    from src.pipeline.run import _build_squad_csv
    df = _build_squad_csv(_make_optimize_result())
    bench = df[~df["is_starter"]].sort_values("bench_order")
    assert bench.iloc[0]["name"] == "Flekken"   # xP 3.0 > 2.0
    assert bench.iloc[0]["bench_order"] == 1

def test_build_squad_csv_captain_flags():
    from src.pipeline.run import _build_squad_csv
    df = _build_squad_csv(_make_optimize_result())
    assert df[df["element"] == 1].iloc[0]["is_captain"] is True
    assert df[df["element"] == 2].iloc[0]["is_vice_captain"] is True
    assert df[df["element"] == 3].iloc[0]["is_captain"] is False
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/test_run_squad_csv.py -v
```

Expected: FAIL (ImportError — `_build_squad_csv` not defined)

- [ ] **Step 3: Add _build_squad_csv to run.py**

Add near top of module-level helpers in `src/pipeline/run.py` (after imports, before phase functions):

```python
def _build_squad_csv(result: dict) -> pd.DataFrame:
    """Assemble combined XI+bench DataFrame for optimal_squad.csv / recommended_squad.csv."""
    xi = result["xi"].copy()
    xi["is_starter"] = True
    xi["bench_order"] = pd.NA

    bench = result["bench"].copy().sort_values("xP", ascending=False).reset_index(drop=True)
    bench["is_starter"] = False
    bench["bench_order"] = range(1, len(bench) + 1)

    df = pd.concat([xi, bench], ignore_index=True)
    cap_el = result["captain"]["element"]
    vc_el = result["vice_captain"]["element"]
    df["is_captain"] = df["element"] == cap_el
    df["is_vice_captain"] = df["element"] == vc_el

    cols = ["element", "name", "position", "team", "xP",
            "is_starter", "bench_order", "is_captain", "is_vice_captain", "now_cost"]
    return df[[c for c in cols if c in df.columns]]
```

- [ ] **Step 4: Update phase_predict to write to gw_dir**

In `src/pipeline/run.py::phase_predict`, replace the output section. Import `gw_dir` at top of file:

```python
from src.config import (
    ..., gw_dir, snapshot_dir, CURRENT_SEASON, ...
)
```

Replace the output block (currently lines ~388–406):

```python
out_dir = gw_dir(CURRENT_SEASON, target_gw)
out_dir.mkdir(parents=True, exist_ok=True)
gw_label = f"gw{target_gw}" if target_gw else "latest"

pred_path = out_dir / "predictions.csv"
save_full_predictions(predictions, pred_path)
print(f"[predict] Saved full predictions ({len(predictions)} players) to {pred_path}")

try:
    result = optimize_team(predictions)
except Exception as e:
    logger.warning(f"optimize_team failed: {e}")
    empty = pd.DataFrame(columns=["element", "name", "position", "team", "xP", "now_cost"])
    _build_squad_csv({"xi": empty, "bench": empty,
                      "captain": {"element": None}, "vice_captain": {"element": None}}
                    ).to_csv(out_dir / "optimal_squad.csv", index=False)
    print(f"[predict] Optimization infeasible — saved empty CSV for {gw_label}")
    return {"xi": empty, "squad": empty, "captain": None, "vice_captain": None, "total_xp": 0.0}

_build_squad_csv(result).to_csv(out_dir / "optimal_squad.csv", index=False)
```

- [ ] **Step 5: Update phase_recommend to write to gw_dir**

In `src/pipeline/run.py::phase_recommend`, replace output paths:

```python
out_dir = gw_dir(CURRENT_SEASON, target_gw)
out_dir.mkdir(parents=True, exist_ok=True)

# recommend.csv (was: RESULTS_DIR / f"recommend_{gw_label}.csv")
out_path = out_dir / "recommend.csv"
save_recommend_csv(plan, out_path, start_gw=target_gw or 0)

# recommended_squad.csv (was: squad_recommend + xi_recommend pair)
if squad_after_ids:
    squad_rec = predictions[predictions["element"].isin(squad_after_ids)][
        ["element", "name", "position", "team", "now_cost", "xP"]
    ].reset_index(drop=True)
    # Build xi/bench split to use _build_squad_csv
    from src.pipeline.optimize import select_xi
    xi_rec = select_xi(squad_rec)
    bench_rec = squad_rec[~squad_rec["element"].isin(xi_rec["element"])]
    xi_cap = xi_rec.sort_values("xP", ascending=False).iloc[0]
    xi_vc  = xi_rec.sort_values("xP", ascending=False).iloc[1]
    rec_result = {"xi": xi_rec, "bench": bench_rec,
                  "captain": xi_cap, "vice_captain": xi_vc}
    _build_squad_csv(rec_result).to_csv(out_dir / "recommended_squad.csv", index=False)
    print(f"Saved post-transfer squad to {out_dir / 'recommended_squad.csv'}")
```

Also update the `pred_path` read at the start of `phase_recommend` (line ~608):
```python
out_dir = gw_dir(CURRENT_SEASON, target_gw)
pred_path = out_dir / "predictions.csv"
```

And in `phase_post_gw`, update reads (lines ~467, 499, 558):
```python
gw_out_dir = gw_dir(CURRENT_SEASON, gw)
pred_path         = gw_out_dir / "predictions.csv"
squad_rec_path    = gw_out_dir / "recommended_squad.csv"
squad_path        = gw_out_dir / "optimal_squad.csv"
```

Note: the local variable `gw_dir` in `phase_post_gw` (line ~446, referring to the vaastav live data dir) must be renamed to `live_gw_dir` to avoid shadowing the imported `gw_dir` function.

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_run_squad_csv.py tests/test_run_recommend_saves_squad.py -v
```

Expected: PASS (squad tests pass; existing recommend saves tests may need path updates — fix any that reference old flat paths)

- [ ] **Step 7: Run full suite**

```bash
python -m pytest tests/ -q --tb=short 2>&1 | tail -15
```

- [ ] **Step 8: Commit**

```bash
git add src/pipeline/run.py tests/test_run_squad_csv.py
git commit -m "feat: add _build_squad_csv, write optimal_squad/recommended_squad to gw_dir"
```

---

## Task 4: run.py — migrate snapshot paths to snapshot_dir()

**Files:**
- Modify: `src/pipeline/run.py`

All snapshot path constructions in `run.py` currently use `SNAPSHOTS_DIR / f"bootstrap_gw{N}.json"`. This task replaces them all with `snapshot_dir(CURRENT_SEASON, gw) / "bootstrap.json"`.

- [ ] **Step 1: Identify all snapshot path references**

```bash
grep -n "bootstrap_gw\|SNAPSHOTS_DIR\|snapshot_dir" src/pipeline/run.py
```

Note every line number. There are approximately 7 occurrences.

- [ ] **Step 2: Update the snapshot load function (_load_bootstrap_snapshot)**

The function at ~line 90 currently does:
```python
snapshot_dir = SNAPSHOTS_DIR          # ← local var shadows import — rename
path = snapshot_dir / f"bootstrap_gw{target_gw}.json"
snapshots = sorted(snapshot_dir.glob("bootstrap_gw*.json"), reverse=True)
```

Replace with:
```python
from src.config import snapshot_dir as get_snapshot_dir  # avoid name clash
if target_gw:
    path = get_snapshot_dir(CURRENT_SEASON, target_gw) / "bootstrap.json"
    if path.exists():
        # check staleness ...
        return json.load(open(path))
# Fallback: find most recent available snapshot
all_snaps = sorted(
    (SNAPSHOTS_DIR / CURRENT_SEASON).glob("*/bootstrap.json"),
    key=lambda p: int(p.parent.name.lstrip("gw")),
    reverse=True,
)
```

- [ ] **Step 3: Update snapshot write paths**

Replace all `SNAPSHOTS_DIR / f"bootstrap_gw{N}.json"` write calls (lines ~151, 211, 228) with:
```python
snap_path = snapshot_dir(CURRENT_SEASON, next_gw) / "bootstrap.json"
snap_path.parent.mkdir(parents=True, exist_ok=True)
with open(snap_path, "w") as f:
    json.dump(bootstrap, f)
```

- [ ] **Step 4: Update daily_bootstrap.yml snapshot path**

In `.github/workflows/daily_bootstrap.yml`, find the step that writes/commits the snapshot and update:
- Write path: `data/snapshots/2025-26/gw${{ steps.check_gw.outputs.current_gw }}/bootstrap.json`
- Commit glob: `data/snapshots/2025-26/` (not `results/snapshots/`)
- `price_changes_latest.txt` path: `data/snapshots/price_changes_latest.txt`

- [ ] **Step 5: Run integration smoke test**

```bash
python -m pytest tests/test_integration_replay.py -v -k "gw31 or gw30" 2>&1 | tail -20
```

Expected: PASS (replay tests use fixture snapshots which are not affected by SNAPSHOTS_DIR)

- [ ] **Step 6: Run full suite**

```bash
python -m pytest tests/ -q --tb=short 2>&1 | tail -10
```

- [ ] **Step 7: Commit**

```bash
git add src/pipeline/run.py .github/workflows/daily_bootstrap.yml
git commit -m "feat: migrate snapshot paths to snapshot_dir() under data/snapshots"
```

---

## Task 5: run.py — _fetch_actual_transfers + phase_post_gw writes

**Files:**
- Modify: `src/pipeline/run.py`
- Modify: `tests/test_run_squad_csv.py`

- [ ] **Step 1: Write failing test for _fetch_actual_transfers**

```python
# tests/test_run_squad_csv.py — add to existing file
from unittest.mock import patch, MagicMock

def test_fetch_actual_transfers_filters_by_gw():
    from src.pipeline.run import _fetch_actual_transfers
    api_response = [
        {"element_in": 10, "element_out": 20, "element_in_cost": 85, "element_out_cost": 85,
         "event": 32, "time": "2026-04-10T10:00:00Z"},
        {"element_in": 11, "element_out": 21, "element_in_cost": 60, "element_out_cost": 65,
         "event": 31, "time": "2026-03-20T10:00:00Z"},
    ]
    bootstrap = {"elements": [
        {"id": 10, "web_name": "Saka"},
        {"id": 20, "web_name": "Salah"},
    ]}
    mock_resp = MagicMock()
    mock_resp.json.return_value = api_response
    with patch("src.pipeline.run._api_get_with_retry", return_value=mock_resp):
        result = _fetch_actual_transfers(entry_id=123, gw=32, bootstrap=bootstrap)
    assert len(result) == 1
    assert result[0]["gw"] == 32
    assert result[0]["player_in"] == "Saka"
    assert result[0]["player_out"] == "Salah"
    assert result[0]["hit_taken"] is False
    assert result[0]["transfer_rank"] == 1

def test_fetch_actual_transfers_hit_taken_when_costs_differ():
    from src.pipeline.run import _fetch_actual_transfers
    api_response = [
        {"element_in": 10, "element_out": 20, "element_in_cost": 85, "element_out_cost": 90,
         "event": 32, "time": "2026-04-10T10:00:00Z"},
    ]
    bootstrap = {"elements": [{"id": 10, "web_name": "A"}, {"id": 20, "web_name": "B"}]}
    mock_resp = MagicMock()
    mock_resp.json.return_value = api_response
    with patch("src.pipeline.run._api_get_with_retry", return_value=mock_resp):
        result = _fetch_actual_transfers(entry_id=1, gw=32, bootstrap=bootstrap)
    assert result[0]["hit_taken"] is True
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/test_run_squad_csv.py::test_fetch_actual_transfers_filters_by_gw tests/test_run_squad_csv.py::test_fetch_actual_transfers_hit_taken_when_costs_differ -v
```

Expected: FAIL

- [ ] **Step 3: Add _fetch_actual_transfers to run.py**

```python
def _fetch_actual_transfers(entry_id: int, gw: int, bootstrap: dict) -> list[dict]:
    """Fetch actual transfers for a GW from FPL API; returns list of dicts."""
    el_map = {e["id"]: e.get("web_name", str(e["id"]))
              for e in bootstrap.get("elements", [])}
    resp = _api_get_with_retry(f"{FPL_ENTRY_URL}/{entry_id}/transfers/")
    all_transfers = resp.json()
    gw_transfers = [t for t in all_transfers if t.get("event") == gw]
    gw_transfers.sort(key=lambda t: t.get("time", ""))
    rows = []
    for rank, t in enumerate(gw_transfers, start=1):
        rows.append({
            "gw": gw,
            "player_out": el_map.get(t["element_out"], str(t["element_out"])),
            "player_in":  el_map.get(t["element_in"],  str(t["element_in"])),
            "transfer_rank": rank,
            "actual_pts_gained": None,   # populated after actual_squad.csv written
            "hit_taken": t.get("element_in_cost") != t.get("element_out_cost"),
        })
    return rows
```

- [ ] **Step 4: Update phase_post_gw to write actual_squad.csv and actual_transfers.csv**

In `src/pipeline/run.py::phase_post_gw`, after the section that computes `your_pts` and fetches `entry_picks_data`, add:

```python
# Write actual_squad.csv (must come before actual_transfers to enable pts join)
if entry_picks_data and not live_df.empty:
    actual_pts_map = live_df.set_index("element")["total_points"].to_dict()
    from src.pipeline.analysis import build_actual_squad_csv
    actual_squad_df = build_actual_squad_csv(entry_picks_data, bootstrap, actual_pts_map)
    actual_squad_path = gw_dir(CURRENT_SEASON, gw) / "actual_squad.csv"
    gw_dir(CURRENT_SEASON, gw).mkdir(parents=True, exist_ok=True)
    actual_squad_df.to_csv(actual_squad_path, index=False)
    print(f"[post-gw] Saved actual_squad.csv ({len(actual_squad_df)} players)")

    # Write/append actual_transfers.csv
    try:
        transfers = _fetch_actual_transfers(entry_id, gw, bootstrap)
        if transfers:
            # Populate actual_pts_gained from actual_squad
            pts_by_name = actual_squad_df.set_index("name")["actual_pts"].to_dict()
            for t in transfers:
                t["actual_pts_gained"] = (
                    pts_by_name.get(t["player_in"], None) or 0
                ) - (pts_by_name.get(t["player_out"], None) or 0) \
                    if t["player_in"] in pts_by_name and t["player_out"] in pts_by_name \
                    else None
            transfers_path = gw_dir(CURRENT_SEASON, gw).parent / "actual_transfers.csv"
            transfers_df = pd.DataFrame(transfers)
            if transfers_path.exists():
                transfers_df.to_csv(transfers_path, mode="a", header=False, index=False)
            else:
                transfers_df.to_csv(transfers_path, index=False)
            print(f"[post-gw] Appended {len(transfers)} transfer(s) to actual_transfers.csv")
    except Exception as e:
        logger.warning(f"Could not fetch actual transfers: {e}")
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_run_squad_csv.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/run.py tests/test_run_squad_csv.py
git commit -m "feat: add _fetch_actual_transfers, write actual_squad.csv + actual_transfers.csv in post-gw"
```

---

## Task 6: generate_reports.py

**Files:**
- Create: `scripts/generate_reports.py`
- Create: `tests/test_generate_reports.py`

- [ ] **Step 1: Write failing tests for chart data preparation**

```python
# tests/test_generate_reports.py
import pandas as pd
import pytest

@pytest.fixture
def sample_accuracy_log(tmp_path):
    df = pd.DataFrame([
        {"gw": 31, "season": "2025-26", "your_pts": 44, "wildcard_pts": 54, "recommended_pts": 8,
         "your_percentile_rank": 20, "best_score": 109, "avg_score": 38},
        {"gw": 32, "season": "2025-26", "your_pts": 54, "wildcard_pts": 38, "recommended_pts": 7,
         "your_percentile_rank": 30, "best_score": 132, "avg_score": 46},
    ])
    p = tmp_path / "accuracy_log.csv"
    df.to_csv(p, index=False)
    return p

def test_load_accuracy_log_sorted_by_season_gw(sample_accuracy_log):
    from scripts.generate_reports import load_accuracy_log
    df = load_accuracy_log(sample_accuracy_log, from_gw=31)
    assert list(df["gw"]) == [31, 32]

def test_load_accuracy_log_filters_from_gw(sample_accuracy_log):
    from scripts.generate_reports import load_accuracy_log
    df = load_accuracy_log(sample_accuracy_log, from_gw=32)
    assert list(df["gw"]) == [32]

def test_estimate_rank_percentile_midpoint():
    from scripts.generate_reports import estimate_rank_percentile
    # 50% between avg_score(38→50%) and best_score(109→0.001%)
    pct = estimate_rank_percentile(score=73, best_score=109, avg_score=38)
    assert 0.001 < pct < 50.0

def test_estimate_rank_percentile_at_avg():
    from scripts.generate_reports import estimate_rank_percentile
    pct = estimate_rank_percentile(score=38, best_score=109, avg_score=38)
    assert abs(pct - 50.0) < 0.1

def test_estimate_rank_percentile_above_best():
    from scripts.generate_reports import estimate_rank_percentile
    pct = estimate_rank_percentile(score=150, best_score=109, avg_score=38)
    assert pct <= 0.001
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/test_generate_reports.py -v
```

Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Create scripts/generate_reports.py**

```python
"""Generate FPL performance reports as PNG charts."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = Path("results")
REPORTS_DIR = RESULTS_DIR / "reports"


def load_accuracy_log(path: Path, from_gw: int = 31) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "season" not in df.columns:
        df["season"] = "2025-26"
    df = df[df["gw"] >= from_gw].sort_values(["season", "gw"]).reset_index(drop=True)
    return df


def estimate_rank_percentile(score: float, best_score: float, avg_score: float) -> float:
    """Linear interpolation in points-space: best_score→0.001%, avg_score→50%, 0→100%."""
    anchors_pts = [0, avg_score, best_score]
    anchors_pct = [100.0, 50.0, 0.001]
    pct = float(np.interp(score, anchors_pts, anchors_pct))
    return max(0.001, min(100.0, pct))


def _decision_impact(accuracy_df: pd.DataFrame, season: str) -> dict[int, float]:
    """Return {gw: actual_pts_gained - recommended_gain} for decision impact panel."""
    transfers_path = RESULTS_DIR / season / "actual_transfers.csv"
    if not transfers_path.exists():
        return {}
    transfers = pd.read_csv(transfers_path)

    impact = {}
    for gw, gw_df in accuracy_df.groupby("gw"):
        rec_path = RESULTS_DIR / season / f"gw{gw}" / "recommend.csv"
        rec_gain = 0.0
        if rec_path.exists():
            rec = pd.read_csv(rec_path)
            rec_gw = rec[rec["gw"] == gw]
            rec_gain = float((rec_gw["xp_in"] - rec_gw["xp_out"]).sum()) if not rec_gw.empty else 0.0

        gw_transfers = transfers[transfers["gw"] == gw]
        actual_gain = float(gw_transfers["actual_pts_gained"].sum()) if not gw_transfers.empty else 0.0
        impact[int(gw)] = actual_gain - rec_gain
    return impact


def plot_gw_chart(accuracy_df: pd.DataFrame, out_path: Path) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_bars, ax_impact) = plt.subplots(
        2, 1, figsize=(max(10, len(accuracy_df) * 1.2), 10),
        gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True
    )

    x = np.arange(len(accuracy_df))
    width = 0.25
    colours = {"My team": "#4472C4", "Optimal": "#70AD47", "Recommended": "#ED7D31"}

    for i, (col, label, colour) in enumerate([
        ("your_pts",        "My team",      colours["My team"]),
        ("wildcard_pts",    "Optimal",      colours["Optimal"]),
        ("recommended_pts", "Recommended",  colours["Recommended"]),
    ]):
        vals = accuracy_df[col].fillna(0).tolist()
        bars = ax_bars.bar(x + (i - 1) * width, vals, width, label=label, color=colour)
        for bar, row, val in zip(bars, accuracy_df.itertuples(), vals):
            if val == 0:
                continue
            best = row.best_score if hasattr(row, "best_score") and pd.notna(row.best_score) else None
            avg  = row.avg_score  if hasattr(row, "avg_score")  and pd.notna(row.avg_score)  else None
            if col == "your_pts" and pd.notna(getattr(row, "your_percentile_rank", None)):
                pct_label = f"top{row.your_percentile_rank:.0f}%"
            elif best and avg:
                pct = estimate_rank_percentile(val, best, avg)
                pct_label = f"top{pct:.1f}%" if pct >= 0.1 else f"top{pct:.3f}%"
            else:
                pct_label = ""
            ax_bars.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.5,
                         f"{int(val)}pts\n({pct_label})" if pct_label else f"{int(val)}pts",
                         ha="center", va="bottom", fontsize=7)

    # Season boundary dividers
    seasons = accuracy_df["season"].tolist()
    for idx in range(1, len(seasons)):
        if seasons[idx] != seasons[idx - 1]:
            ax_bars.axvline(x=idx - 0.5, color="gray", linestyle="--", linewidth=0.8)

    # X-axis labels with season annotations
    tick_labels = [f"GW{r.gw}" for r in accuracy_df.itertuples()]
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax_bars.set_ylabel("Points scored")
    ax_bars.set_title("GW Performance: My Team vs Optimal vs Recommended")
    ax_bars.legend()

    # Add season group labels below x-axis
    season_groups: dict[str, list[int]] = {}
    for idx, s in enumerate(seasons):
        season_groups.setdefault(s, []).append(idx)
    for season, idxs in season_groups.items():
        mid = np.mean(idxs)
        ax_bars.annotate(f"── {season} ──", xy=(mid, 0), xycoords=("data", "axes fraction"),
                         ha="center", va="top", fontsize=8, color="gray",
                         xytext=(0, -30), textcoords="offset points")

    # Bottom panel: decision impact
    all_gws = accuracy_df["gw"].tolist()
    season_col = accuracy_df["season"].tolist()
    impact = {}
    for s in accuracy_df["season"].unique():
        impact.update(_decision_impact(accuracy_df[accuracy_df["season"] == s], s))
    impact_vals = [impact.get(gw, 0.0) for gw in all_gws]
    bar_colours = ["#70AD47" if v >= 0 else "#FF0000" for v in impact_vals]
    ax_impact.bar(x, impact_vals, color=bar_colours)
    ax_impact.axhline(0, color="black", linewidth=0.8)
    ax_impact.set_xticks(x)
    ax_impact.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax_impact.set_ylabel("Transfer impact (pts)")
    ax_impact.set_title("Decision Impact vs Recommendation")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[reports] Saved {out_path}")


def plot_season_chart(accuracy_df: pd.DataFrame, out_path: Path) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(accuracy_df) * 0.8), 6), constrained_layout=True)

    for col, label, colour in [
        ("your_pts",        "My team",      "#4472C4"),
        ("wildcard_pts",    "Optimal",      "#70AD47"),
        ("recommended_pts", "Recommended",  "#ED7D31"),
    ]:
        cumulative = accuracy_df[col].fillna(0).cumsum()
        ax.plot(range(len(accuracy_df)), cumulative, marker="o", label=label, color=colour)

    tick_labels = [f"GW{r.gw}" for r in accuracy_df.itertuples()]
    ax.set_xticks(range(len(accuracy_df)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cumulative points")
    ax.set_title("Cumulative Season Performance")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[reports] Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-gw", type=int, default=31)
    parser.add_argument("--accuracy-log", type=Path, default=RESULTS_DIR / "accuracy_log.csv")
    args = parser.parse_args()

    if not args.accuracy_log.exists():
        print(f"[reports] {args.accuracy_log} not found — nothing to plot")
        return

    df = load_accuracy_log(args.accuracy_log, from_gw=args.from_gw)
    if df.empty:
        print(f"[reports] No data from GW{args.from_gw} onwards")
        return

    plot_gw_chart(df, REPORTS_DIR / "rank_comparison_gw.png")
    plot_season_chart(df, REPORTS_DIR / "rank_comparison_season.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_generate_reports.py -v
```

Expected: PASS

- [ ] **Step 5: Smoke test the script against real data**

```bash
python scripts/generate_reports.py --from-gw 31
```

Expected: `results/reports/rank_comparison_gw.png` and `rank_comparison_season.png` created. Open both to verify they render correctly.

- [ ] **Step 6: Add generate_reports call to phase_post_gw**

At the end of `phase_post_gw` in `src/pipeline/run.py`, after appending transfers:

```python
# Regenerate reports
try:
    import subprocess
    subprocess.run(
        ["python", "scripts/generate_reports.py", "--from-gw", "31"],
        check=True
    )
except Exception as e:
    logger.warning(f"generate_reports failed (non-fatal): {e}")
```

- [ ] **Step 7: Update CI workflow**

In `.github/workflows/daily_bootstrap.yml`, after the post-gw step (gated on `gw_finished == 'true'`), add:

```yaml
- name: Generate performance reports
  if: steps.check_gw_finished.outputs.gw_finished == 'true'
  run: python scripts/generate_reports.py --from-gw 31

- name: Commit results and reports
  if: steps.check_gw_finished.outputs.gw_finished == 'true'
  run: |
    git add results/2025-26/ results/accuracy_log.csv results/reports/ data/snapshots/2025-26/
    git diff --cached --quiet || git commit -m "chore: post-gw results and reports GW${{ steps.check_gw_finished.outputs.current_gw }}"
```

- [ ] **Step 8: Commit**

```bash
git add scripts/generate_reports.py tests/test_generate_reports.py src/pipeline/run.py .github/workflows/daily_bootstrap.yml
git commit -m "feat: add generate_reports.py with GW and season rank charts"
```

---

## Task 7: Migration script — reorganise existing files

**Files:**
- Create: `scripts/migrate_results.py`

This script runs once locally and is deleted after use.

- [ ] **Step 1: Create scripts/migrate_results.py**

```python
"""One-off migration: reorganise results/ and snapshots/ to new season/GW structure.
Run once, then delete this script.
"""
from pathlib import Path
import shutil
import pandas as pd

RESULTS = Path("results")
DATA    = Path("data")
GWS = [30, 31, 32, 33]
SEASON = "2025-26"


def move(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"  SKIP (not found): {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"  MOVE {src} → {dst}")


def merge_squad(squad_path: Path, xi_path: Path) -> pd.DataFrame:
    squad = pd.read_csv(squad_path)
    xi_elements = set(pd.read_csv(xi_path)["element"].tolist()) if xi_path.exists() else set()
    squad["is_starter"] = squad["element"].isin(xi_elements)
    bench = squad[~squad["is_starter"]].sort_values("xP", ascending=False).reset_index(drop=True)
    bench_order = {row["element"]: i + 1 for i, row in bench.iterrows()}
    squad["bench_order"] = squad["element"].map(bench_order)
    squad["is_captain"] = None
    squad["is_vice_captain"] = None
    return squad[["element", "name", "position", "team", "xP",
                  "is_starter", "bench_order", "is_captain", "is_vice_captain", "now_cost"]]


def main():
    for gw in GWS:
        gw_dir = RESULTS / SEASON / f"gw{gw}"
        gw_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- GW{gw} ---")

        move(RESULTS / f"predictions_gw{gw}.csv", gw_dir / "predictions.csv")
        move(RESULTS / f"recommend_gw{gw}.csv",   gw_dir / "recommend.csv")

        squad_src = RESULTS / f"squad_gw{gw}.csv"
        xi_src    = RESULTS / f"xi_gw{gw}.csv"
        if squad_src.exists():
            merged = merge_squad(squad_src, xi_src)
            merged.to_csv(gw_dir / "optimal_squad.csv", index=False)
            print(f"  MERGE {squad_src} + {xi_src} → {gw_dir / 'optimal_squad.csv'}")
            squad_src.unlink(missing_ok=True)
            xi_src.unlink(missing_ok=True)

        rec_squad_src = RESULTS / f"squad_recommend_gw{gw}.csv"
        rec_xi_src    = RESULTS / f"xi_recommend_gw{gw}.csv"
        if rec_squad_src.exists():
            merged_rec = merge_squad(rec_squad_src, rec_xi_src)
            merged_rec.to_csv(gw_dir / "recommended_squad.csv", index=False)
            print(f"  MERGE {rec_squad_src} + {rec_xi_src} → {gw_dir / 'recommended_squad.csv'}")
            rec_squad_src.unlink(missing_ok=True)
            rec_xi_src.unlink(missing_ok=True)

        # Move bootstrap snapshot
        snap_src = RESULTS / "snapshots" / f"bootstrap_gw{gw}.json"
        snap_dst = DATA / "snapshots" / SEASON / f"gw{gw}" / "bootstrap.json"
        move(snap_src, snap_dst)

    # Move price_changes_latest.txt
    pct_src = RESULTS / "snapshots" / "price_changes_latest.txt"
    pct_dst = DATA / "snapshots" / "price_changes_latest.txt"
    move(pct_src, pct_dst)

    # Delete signal_unresolved.csv
    sig = RESULTS / "signal_unresolved.csv"
    if sig.exists():
        sig.unlink()
        print(f"\nDELETED {sig}")

    # Backfill season column in accuracy_log.csv
    log = RESULTS / "accuracy_log.csv"
    if log.exists():
        df = pd.read_csv(log)
        if "season" not in df.columns:
            df.insert(1, "season", SEASON)
            df.to_csv(log, index=False)
            print(f"\nBackfilled season='{SEASON}' in {log}")

    # Remove now-empty snapshots dir if empty
    snap_dir = RESULTS / "snapshots"
    if snap_dir.exists() and not any(snap_dir.iterdir()):
        snap_dir.rmdir()
        print(f"Removed empty {snap_dir}")

    print("\nMigration complete. Verify contents then delete this script.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the migration script**

```bash
python scripts/migrate_results.py
```

Verify output: no unexpected SKIPs. Check `results/2025-26/gw31/` contains the expected files.

- [ ] **Step 3: Verify new structure**

```bash
ls results/2025-26/gw31/ results/2025-26/gw32/ results/2025-26/gw33/
ls data/snapshots/2025-26/
head -2 results/accuracy_log.csv   # confirm season column present
```

- [ ] **Step 4: Commit migrated files then delete script**

```bash
git add results/ data/snapshots/
git commit -m "chore: migrate results to 2025-26/gw{N}/ structure, move snapshots to data/"
git rm scripts/migrate_results.py
git commit -m "chore: remove one-off migrate_results.py"
```

---

## Task 8: Backfill actual_squad.csv + actual_transfers.csv for GW31–33

**Files:**
- Create: `scripts/backfill_actuals.py`

Requires `user_config.yaml` with a valid `entry_id`.

- [ ] **Step 1: Create scripts/backfill_actuals.py**

```python
"""One-off backfill: create actual_squad.csv + actual_transfers.csv for GW31-33.
Run once after migrate_results.py, then delete this script.
Requires: user_config.yaml present with teams.default.entry_id set.
"""
from pathlib import Path
import pandas as pd

from src.config import CURRENT_SEASON, DATA_DIR, gw_dir
from src.pipeline.config import load_user_config
from src.pipeline.fetch import _api_get_with_retry
from src.pipeline.analysis import build_actual_squad_csv
from src.pipeline.run import _fetch_actual_transfers
from src.config import FPL_ENTRY_URL, FPL_EVENT_URL

RESULTS = Path("results")
BACKFILL_GWS = [31, 32, 33]


def main():
    cfg = load_user_config()
    entry_id = cfg["teams"]["default"]["entry_id"]
    print(f"Backfilling for entry_id={entry_id}, GWs {BACKFILL_GWS}")

    transfers_path = RESULTS / CURRENT_SEASON / "actual_transfers.csv"
    all_transfers = []

    for gw in BACKFILL_GWS:
        print(f"\n--- GW{gw} ---")
        out_dir = gw_dir(CURRENT_SEASON, gw)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load bootstrap for this GW (already migrated)
        snap_path = DATA_DIR / "snapshots" / CURRENT_SEASON / f"gw{gw}" / "bootstrap.json"
        if not snap_path.exists():
            print(f"  SKIP: bootstrap not found at {snap_path}")
            continue
        import json
        with open(snap_path) as f:
            bootstrap = json.load(f)

        # Fetch picks
        try:
            picks_data = _api_get_with_retry(
                f"{FPL_ENTRY_URL}/{entry_id}/event/{gw}/picks/"
            ).json()
        except Exception as e:
            print(f"  SKIP picks: {e}")
            continue

        # Fetch live GW points
        try:
            live_resp = _api_get_with_retry(f"{FPL_EVENT_URL}/{gw}/live/").json()
            actual_pts_map = {
                el["id"]: el["stats"]["total_points"]
                for el in live_resp.get("elements", [])
            }
        except Exception as e:
            print(f"  SKIP live data: {e}")
            actual_pts_map = {}

        # Write actual_squad.csv
        actual_df = build_actual_squad_csv(picks_data, bootstrap, actual_pts_map)
        actual_path = out_dir / "actual_squad.csv"
        actual_df.to_csv(actual_path, index=False)
        print(f"  Wrote {actual_path} ({len(actual_df)} players)")

        # Fetch actual transfers
        try:
            transfers = _fetch_actual_transfers(entry_id, gw, bootstrap)
            pts_by_name = actual_df.set_index("name")["actual_pts"].to_dict()
            for t in transfers:
                t["actual_pts_gained"] = (
                    (pts_by_name.get(t["player_in"], 0) or 0)
                    - (pts_by_name.get(t["player_out"], 0) or 0)
                ) if t["player_in"] in pts_by_name and t["player_out"] in pts_by_name else None
            all_transfers.extend(transfers)
            print(f"  Fetched {len(transfers)} transfer(s)")
        except Exception as e:
            print(f"  Could not fetch transfers: {e}")

    # Write combined actual_transfers.csv
    if all_transfers:
        df = pd.DataFrame(all_transfers)
        df.to_csv(transfers_path, index=False)
        print(f"\nWrote {transfers_path} ({len(df)} rows)")

    print("\nBackfill complete. Delete this script.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the backfill script**

```bash
python scripts/backfill_actuals.py
```

Expected: `actual_squad.csv` created in each of `results/2025-26/gw31/`, `gw32/`, `gw33/`. `actual_transfers.csv` created at `results/2025-26/actual_transfers.csv`.

- [ ] **Step 3: Verify output**

```bash
head -3 results/2025-26/gw31/actual_squad.csv
cat results/2025-26/actual_transfers.csv
```

- [ ] **Step 4: Regenerate reports with backfilled data**

```bash
python scripts/generate_reports.py --from-gw 31
```

Open `results/reports/rank_comparison_gw.png` and `rank_comparison_season.png` to verify GW31–33 render correctly with decision impact bars.

- [ ] **Step 5: Commit backfilled files, delete script**

```bash
git add results/2025-26/
git commit -m "chore: backfill actual_squad and actual_transfers for GW31-33"
git rm scripts/backfill_actuals.py
git commit -m "chore: remove one-off backfill_actuals.py"
```

---

## Task 9: Final wiring + full test run

**Files:**
- Modify: `tests/test_run_recommend_saves_squad.py` (update any paths referencing old flat structure)

- [ ] **Step 1: Find and fix any tests with old flat paths**

```bash
grep -rn "squad_gw\|xi_gw\|squad_recommend\|xi_recommend\|predict_gw\|bootstrap_gw" tests/ | grep -v ".pyc"
```

For each occurrence, update to use `gw_dir(CURRENT_SEASON, N) / "filename.csv"` or `snapshot_dir(CURRENT_SEASON, N) / "bootstrap.json"`.

- [ ] **Step 2: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all tests pass (or identify any remaining path-related failures and fix them).

- [ ] **Step 3: Smoke test full pipeline on GW33 data**

```bash
python -m src.pipeline.run predict --gw 34   # dry-run predict to verify new paths written
ls results/2025-26/gw34/
```

Expected: `predictions.csv` and `optimal_squad.csv` appear under `results/2025-26/gw34/`.

- [ ] **Step 4: Final commit**

```bash
git add tests/
git commit -m "test: update path references to new season/gw structure"
```

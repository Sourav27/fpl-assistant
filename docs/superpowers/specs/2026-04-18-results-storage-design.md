# Results Storage & Performance Visibility — Design Spec

**Date:** 2026-04-18
**Status:** Approved
**Scope:** Reorganise `results/` into a season/GW folder structure; add weekly rank-comparison plots and transfer decision log. Standalone, lightweight — no web server. Track F (full dashboard) is deferred to next season.

---

## Objectives

1. Clean, navigable `results/` folder organised by season and gameweek; snapshots moved to `data/`
2. Per-GW and cumulative season rank-comparison PNGs across three teams (actual, optimal, recommended)
3. Transfer decision log: what was recommended vs what was done, with point impact
4. Backfill `actual_squad.csv` and `actual_transfers.csv` for GW31–33 via FPL API

---

## Folder Structure

```
data/
  Fantasy-Premier-League/    ← vaastav dataset (unchanged)
  snapshots/
    2025-26/
      gw31/
        bootstrap.json       ← FPL API snapshot used for that GW's pipeline run
      gw32/  …
    price_changes_latest.txt ← always current; stays at data/snapshots/ root

results/
  2025-26/
    gw31/
      predictions.csv        ← xP for all ~600 players (model input → ranked picks)
      optimal_squad.csv      ← 15-player optimizer squad: XI + bench, xP only
      recommend.csv          ← suggested transfers (in/out, xP delta, hit cost, bank)
      recommended_squad.csv  ← 15-player post-transfer squad: XI + bench, xP only
      actual_squad.csv       ← post-gw only: real fielded squad + actual_pts (sacred — never overwritten by predict)
    gw32/  …
    gw33/  …
    actual_transfers.csv     ← season-level actual transfer log (one row per transfer)
  accuracy_log.csv           ← cross-season, stays at RESULTS_DIR root
  reports/
    rank_comparison_gw.png        ← all GWs across all seasons: grouped bars, grows each post-gw
    rank_comparison_season.png    ← cumulative season: running total rank, regenerated each post-gw
```

**Key invariant:** `actual_squad.csv` is written only by `post-gw`. The `predict` and `recommend` phases write only `predictions.csv`, `optimal_squad.csv`, `recommend.csv`, and `recommended_squad.csv`. Re-running predict for model testing never touches `actual_squad.csv`.

**Why snapshots in `data/`:** Snapshots are input data (FPL API state at deadline time), not pipeline outputs. They can be purged when the vaastav dataset is updated to cover the season. Results are pipeline outputs and should not mix with inputs.

**Why `actual_transfers.csv` inside season:** Transfer decisions are season-specific context; no cross-season transfer comparison is planned. `accuracy_log.csv` stays at root because model accuracy metrics are compared across seasons.

---

## File Schemas

### `optimal_squad.csv` / `recommended_squad.csv`
Written by predict/recommend phase. Contains xP predictions only. No actual points.

Assembly source: `optimize_team()` return dict — `result["xi"]`, `result["bench"]`, `result["captain"]`, `result["vice_captain"]`. A new helper `_build_squad_csv(result: dict) -> pd.DataFrame` in `run.py` performs this:
- XI players (`result["xi"]`): `is_starter=True`, `bench_order=null`
- Bench players (`result["bench"]`): `is_starter=False`, `bench_order` = rank by xP **descending** (1 = highest-xP bench player). Display-only — does not represent FPL auto-sub priority order.
- `is_captain = (element == result["captain"]["element"])`
- `is_vice_captain = (element == result["vice_captain"]["element"])`
- Concat xi rows then bench rows

| Column | Type | Description |
|--------|------|-------------|
| element | int | FPL element ID (for joins) |
| name | str | Player display name |
| position | str | GK / DEF / MID / FWD |
| team | str | Club name |
| xP | float | Predicted points this GW |
| is_starter | bool | True = XI, False = bench |
| bench_order | int | 1–4 for bench (display rank by xP desc), null for starters |
| is_captain | bool | |
| is_vice_captain | bool | |
| now_cost | float | Price in £M |

### `actual_squad.csv`
Written by `run.py::phase_post_gw` only. Never written by predict or recommend.

Source: FPL picks API `/api/entry/{entry_id}/event/{gw}/picks/` returns `element, position, multiplier, is_captain, is_vice_captain`. Names, team, and `now_cost` joined from bootstrap `elements` list for that GW (`data/snapshots/2025-26/gw{N}/bootstrap.json`). `actual_pts` sourced from per-player event stats fetched in `phase_post_gw`.

Written **before** `actual_transfers.csv` in `phase_post_gw` so the `actual_pts_gained` join can read it.

| Column | Type | Description |
|--------|------|-------------|
| element | int | FPL element ID (join key for transfers) |
| name | str | Player display name |
| position | str | GK / DEF / MID / FWD |
| team | str | Club name |
| actual_pts | int | Points scored this GW (incl. captain multiplier, auto-subs) |
| is_starter | bool | |
| bench_order | int | 1–4 for bench (display only), null for starters |
| is_captain | bool | |
| is_vice_captain | bool | |
| now_cost | float | Price in £M |

### `actual_transfers.csv` (season-level, inside `results/2025-26/`)
One row per transfer made. Written/appended by `run.py::phase_post_gw` (after `actual_squad.csv`).

**Fetch:** new `_fetch_actual_transfers(entry_id: int, gw: int, bootstrap: dict) -> list[dict]` in `run.py`, calling `/api/entry/{entry_id}/transfers/` via existing `fetch.py` retry wrapper. Response is the full season transfer list with fields `element_in, element_out, element_in_cost, element_out_cost, event, time`. Filter to `event == gw`. Sort by `time` (ISO timestamp string) ascending within the filtered set → `transfer_rank` (1-based). Name lookup via bootstrap `elements` list; fall back to `str(element_id)` if not found.

**`actual_pts_gained`:** join on `element` against `actual_squad.csv` written earlier in the same `phase_post_gw` call. `player_in actual_pts − player_out actual_pts`. Null if either element absent.

**`hit_taken`:** `element_in_cost != element_out_cost` (FPL API encodes hit transfers with differing costs).

| Column | Type | Description |
|--------|------|-------------|
| gw | int | Gameweek |
| player_out | str | Player transferred out (display name) |
| player_in | str | Player transferred in (display name) |
| transfer_rank | int | Order within GW (1-based, sorted by API `time` asc) |
| actual_pts_gained | float | player_in actual_pts − player_out actual_pts (null if unavailable) |
| hit_taken | bool | True if this transfer cost -4 pts |

**Decision impact in reports:** `generate_reports.py` joins `actual_transfers.csv` with per-GW `recommend.csv` on `(gw, player_out, player_in)` (exact name match). Matched → recommended_gain = `xp_in − xp_out`. Unmatched → recommended_gain = 0. Per-GW bar: `sum(actual_pts_gained) − sum(recommended_gain)`.

### `accuracy_log.csv` — schema change
Add `season` column (string, e.g. `"2025-26"`). Populated by `append_accuracy_log` going forward using `CURRENT_SEASON`. Backfilled as `"2025-26"` for existing GW31–33 rows during migration.

This column is the join key that lets `generate_reports.py` distinguish GW31 2025-26 from GW31 2026-27 on the X-axis.

### `recommend.csv` (unchanged schema)
`gw, action, player_out, player_in, price_out, price_in, xp_out, xp_in, hit_cost, bank_after`

---

## Reports

### `reports/rank_comparison_gw.png` — all GWs across seasons (single growing file)

Regenerated each post-gw. Covers GW31 2025-26 onwards; grows as new GWs and seasons are added.

**Layout:** Two panels.

**Top panel — grouped bars per GW:**
X-axis: GW labels (`GW31`, `GW32`, …) grouped by season. Season boundaries marked with a vertical divider line; season label shown as a group annotation below (e.g. `── 2025-26 ──`). Data read from `accuracy_log.csv` ordered by `(season, gw)`.

Y-axis: points scored (higher = better, natural bar direction).

Three bars per GW:
- **My team** → `your_pts`
- **Optimal squad** → `wildcard_pts`
- **Recommended squad** → `recommended_pts`

Data label on each bar: `Xpts\n(topY%)`. Rank percentile derived from two-anchor interpolation: `best_score → 0.001%`, `avg_score → 50%` (linear in points-space). `your_percentile_rank` used directly for "My team" when available. Bars colour-coded: blue = My team, green = Optimal, orange = Recommended.

`top_1k_score` / `top_10k_score` / `top_100k_score` are **not used** — see bug note below.

**Bottom panel — Decision impact (transfer delta per GW):**
Bar per GW: `sum(actual_pts_gained) − sum(recommended_gain)`. Green = beat recommendation; red = underperformed. GW31–33 backfilled via `backfill_actuals.py`.

### `reports/rank_comparison_season.png` — cumulative season

Regenerated each post-gw. Single chart.

**X-axis:** GW number (GW31+). **Y-axis:** approximate season rank percentile (log scale).

Three lines: cumulative total points for each team (summing GW scores from GW31 to current GW), converted to rank percentile using the same two-anchor interpolation (`best_score`, `avg_score` averaged across included GWs, with `ranked_count` from latest GW).

### Bug: `top_N_score` values are unreliable

`fetch_gw_benchmarks` currently fetches standings page N/50 by **season rank** and reads that player's GW score — but season rank ≠ GW rank. A player ranked 100,000th by season total can outscore one ranked 10,000th in a single GW. This is why `top_100k_score` (68) > `top_10k_score` (43) in GW32.

**Fix (deferred to a separate task):** Replace the standings-pagination approach with a proper GW-rank lookup. Options: (a) use FPL's `leagues-classic/{overall_league_id}/standings/` sorted by `event_total` for that GW (requires checking if `&ordering=-event_total` is supported), or (b) fetch a large sample and sort locally. Until fixed, `top_1k/10k/100k` columns in `accuracy_log.csv` should be treated as unreliable and are excluded from rank_comparison interpolation.

---

## New & Changed Components

### `scripts/migrate_results.py` (one-off, deleted after use)

| Existing file | New path | Notes |
|---|---|---|
| `results/predictions_gw{N}.csv` | `results/2025-26/gw{N}/predictions.csv` | Direct move |
| `results/squad_gw{N}.csv` + `results/xi_gw{N}.csv` | `results/2025-26/gw{N}/optimal_squad.csv` | Merge (below) |
| `results/recommend_gw{N}.csv` | `results/2025-26/gw{N}/recommend.csv` | Direct move |
| `results/squad_recommend_gw{N}.csv` + `results/xi_recommend_gw{N}.csv` | `results/2025-26/gw{N}/recommended_squad.csv` | Merge (below) |
| `results/snapshots/bootstrap_gw{N}.json` | `data/snapshots/2025-26/gw{N}/bootstrap.json` | Move to data/ |

**Merge logic (squad + xi → combined CSV):**
Existing squad columns: `element, name, position, team, now_cost, xP, raw_xP`. XI = subset (starters). `is_starter = element in xi["element"].values`. Bench: `bench_order` = rank by xP desc (1–4). `is_captain`/`is_vice_captain` = null for GW30–33. Drop `raw_xP`. Keep `element`.

**Also:** Delete `results/signal_unresolved.csv`. Backfill `accuracy_log.csv`: add `season` column with value `"2025-26"` for all existing rows (GW31–33).

### `scripts/backfill_actuals.py` (one-off for GW31–33, deleted after use)
New script. Uses FPL API to retroactively create `actual_squad.csv` and `actual_transfers.csv` for GW31–33.
- For each GW in [31, 32, 33]: fetch `/api/entry/{entry_id}/event/{gw}/picks/` + transfers endpoint + bootstrap snapshot (already migrated to `data/snapshots/`)
- Write `results/2025-26/gw{N}/actual_squad.csv` and append to `results/2025-26/actual_transfers.csv`
- Requires `user_config.yaml` to be present for `entry_id`

### `config.py`
- Add `CURRENT_SEASON = "2025-26"`
- Add `def gw_dir(season: str, gw: int) -> Path: return RESULTS_DIR / season / f"gw{gw}"`
- Change `SNAPSHOTS_DIR = RESULTS_DIR / "snapshots"` → `SNAPSHOTS_DIR = DATA_DIR / "snapshots"` (move from results to data)
- Add `def snapshot_dir(season: str, gw: int) -> Path: return SNAPSHOTS_DIR / season / f"gw{gw}"`
- Confirm `DATA_DIR = BASE_DIR / "data"` exists (add if not)
- Remove `SIGNAL_UNRESOLVED_CSV`

### `src/pipeline/datasources/signals.py`
`log_unresolved_name` called from `ffs.py`, `premierinjuries.py`, `reddit.py` — all pass `name, source, raw_text, timestamp` (no `csv_path`). Keep function signature; replace body with `logging.warning(f"[{source}] Unresolved player: {name!r} — {raw_text[:80]!r}")`. Remove `from src.config import SIGNAL_UNRESOLVED_CSV` from function body. `__init__.py` export unchanged.

### `src/pipeline/run.py`
- Add `_build_squad_csv(result: dict) -> pd.DataFrame`
- Add `_fetch_actual_transfers(entry_id: int, gw: int, bootstrap: dict) -> list[dict]`
- `phase_predict`: write to `gw_dir(CURRENT_SEASON, gw)`; remove old xi/squad flat writes
- `phase_recommend`: write to `gw_dir()`; remove old squad_recommend/xi_recommend writes
- Snapshot read/write: replace `SNAPSHOTS_DIR / f"bootstrap_gw{N}.json"` → `snapshot_dir(CURRENT_SEASON, gw) / "bootstrap.json"`
- Fallback glob: `sorted((SNAPSHOTS_DIR / CURRENT_SEASON).glob("*/bootstrap.json"), key=lambda p: int(p.parent.name.lstrip("gw")), reverse=True)`
- `phase_post_gw`: write `actual_squad.csv` → append `actual_transfers.csv` → call `generate_reports.py`
- `actual_transfers.csv` path: `gw_dir(CURRENT_SEASON, gw).parent / "actual_transfers.csv"` (season root, not per-GW)

### `src/pipeline/analysis.py`
- New `build_actual_squad_csv(entry_picks, bootstrap, actual_pts_by_element) -> pd.DataFrame`
- `append_accuracy_log`: add `season` parameter (default `CURRENT_SEASON`); write as first column after `gw`

### `scripts/generate_reports.py` (new)
```
python scripts/generate_reports.py [--from-gw 31]
```
- Reads `results/accuracy_log.csv` + `results/2025-26/actual_transfers.csv` + per-GW `recommend.csv`
- Writes `results/reports/rank_comparison_gw.png` (all GWs across seasons, regenerated in full)
- Writes `results/reports/rank_comparison_season.png`

### `.github/workflows/daily_bootstrap.yml`
- Snapshot write: `data/snapshots/2025-26/gw{N}/bootstrap.json`
- Snapshot commit glob: `data/snapshots/2025-26/`
- After post-gw (gated on `gw_finished == 'true'`): `python scripts/generate_reports.py --from-gw 31`
- Results commit glob: add `results/reports/`
- `price_changes_latest.txt` at `data/snapshots/price_changes_latest.txt` (path updated to follow SNAPSHOTS_DIR move)

---

## Migration Plan

1. Run `scripts/migrate_results.py` — reorganises results/, moves snapshots to data/
2. Run `scripts/backfill_actuals.py` — creates actual_squad.csv + actual_transfers.csv for GW31–33
3. Verify `results/2025-26/gw{31..33}/` and `data/snapshots/2025-26/` contents
4. Commit reorganised files to git
5. Delete both scripts, commit deletion
6. Pipeline writes new paths from GW34+ automatically

---

## Out of Scope

- Interactive web dashboard (Track F, next season)
- Fix for `top_N_score` unreliability in `fetch_gw_benchmarks` (deferred, tracked as separate bug)
- `raw_xP` dropped from squad CSVs (still in `predictions.csv` for model evaluation)

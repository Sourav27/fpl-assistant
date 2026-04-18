# Results Storage & Performance Visibility — Design Spec

**Date:** 2026-04-18
**Status:** Approved
**Scope:** Reorganise `results/` into a season/GW folder structure; add weekly rank-comparison plot and transfer decision log. Standalone, lightweight — no web server. Track F (full dashboard) is deferred to next season.

---

## Objectives

1. Clean, navigable `results/` folder organised by season and gameweek
2. Weekly PNG plot comparing GW rank across three teams (actual, optimal, recommended)
3. Transfer decision log: what was recommended vs what was done, with point impact

---

## Folder Structure

```
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
  snapshots/
    2025-26/
      gw31/
        bootstrap.json       ← FPL API snapshot used for that GW's pipeline run
      gw32/  …
    price_changes_latest.txt ← always current; path = SNAPSHOTS_DIR / "price_changes_latest.txt" (unchanged)
  accuracy_log.csv           ← season-level, stays at RESULTS_DIR root (path unchanged)
  actual_transfers.csv       ← season-level actual transfer history (new)
  reports/
    rank_comparison.png      ← regenerated each post-gw
```

**Key invariant:** `actual_squad.csv` is written only by `post-gw`. The `predict` and `recommend` phases write only `predictions.csv`, `optimal_squad.csv`, `recommend.csv`, and `recommended_squad.csv`. Re-running predict for model testing never touches `actual_squad.csv`.

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

Source: FPL picks API `/api/entry/{entry_id}/event/{gw}/picks/` returns `element, position, multiplier, is_captain, is_vice_captain`. Names, team, and `now_cost` joined from bootstrap `elements` list for that GW. `actual_pts` sourced from the per-player event stats already fetched in `phase_post_gw`.

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

### `actual_transfers.csv`
Season-level. One row per transfer made. Written/appended by `run.py::phase_post_gw` (after `actual_squad.csv`).

**Fetch:** new `_fetch_actual_transfers(entry_id: int, gw: int, bootstrap: dict) -> list[dict]` in `run.py`, calling `/api/entry/{entry_id}/transfers/` via existing `fetch.py` retry wrapper. Response is the full season transfer list with fields `element_in, element_out, element_in_cost, element_out_cost, event, time`. Filter to `event == gw`. Sort by `time` (ISO timestamp string) ascending within the filtered set → `transfer_rank` (1-based). Name lookup via bootstrap `elements` list; fall back to `str(element_id)` if not found.

**`actual_pts_gained`:** join on `element` field against `actual_squad.csv` written earlier in the same `phase_post_gw` call. `player_in actual_pts − player_out actual_pts`. Null if either element is absent from `actual_squad.csv`.

**`hit_taken`:** `element_in_cost != element_out_cost` (FPL API encodes hit transfers with different cost values) — alternatively derive from `event_transfers_cost > 0` in entry history.

| Column | Type | Description |
|--------|------|-------------|
| gw | int | Gameweek |
| player_out | str | Player transferred out (display name) |
| player_in | str | Player transferred in (display name) |
| transfer_rank | int | Order within GW (1-based, sorted by API `time` asc) |
| actual_pts_gained | float | player_in actual_pts − player_out actual_pts (null if unavailable) |
| hit_taken | bool | True if this transfer cost -4 pts |

**Decision impact in `generate_reports.py`:** join `actual_transfers.csv` with per-GW `recommend.csv` on `(gw, player_out, player_in)` (exact name match). Matched → recommended_gain = `xp_in − xp_out` from `recommend.csv`. Unmatched actual transfer → recommended_gain = 0. Bar chart per GW shows `sum(actual_pts_gained) − sum(recommended_gain)`.

### `recommend.csv` (unchanged schema)
| Column | Type | Description |
|--------|------|-------------|
| gw | int | |
| action | str | "transfer" |
| player_out | str | |
| player_in | str | |
| price_out | float | |
| price_in | float | |
| xp_out | float | |
| xp_in | float | |
| hit_cost | int | 0 or 4 |
| bank_after | float | |

---

## `reports/rank_comparison.png`

Two-panel PNG. Regenerated each post-gw. Covers GW31 onwards.

**Top panel — Rank percentile lines:**

X-axis: GW number. Y-axis: approximate rank percentile on log scale (lower = better; labeled "top X%").

Three lines from `accuracy_log.csv`:
- **My team** → `your_pts`; use `your_percentile_rank` directly (already stored)
- **Optimal squad** → `wildcard_pts` (the unconstrained optimizer squad's actual points — semantically identical to "optimal squad" here)
- **Recommended squad** → `recommended_pts`

For Optimal and Recommended lines, rank percentile is derived by linear interpolation in **points-space** between these anchors (then plotted on log-scale Y):

| Score threshold | Rank percentile |
|---|---|
| ≥ `best_score` | 0.001% |
| `top_1k_score` | 0.015% |
| `top_10k_score` | 0.15% |
| `top_100k_score` | 1.5% |
| `avg_score` | 50% |
| 0 pts | 100% |

Interpolation: `numpy.interp(score, thresholds_asc, percentiles_asc)`. Scores above `best_score` clamped to 0.001%. Scores below 0 clamped to 100%.

**Bottom panel — Decision impact bars:**

Bar chart per GW: `sum(actual_pts_gained) − sum(recommended_gain)`. Green = beat or matched recommendation; red = underperformed. Zero baseline. GWs with no `actual_transfers.csv` data (GW30–33 backfill gap) shown as zero-height bars.

---

## New & Changed Components

### `scripts/migrate_results.py` (one-off, deleted after use)

Exact filename patterns for GW30–33:

| Existing file | New path | Notes |
|---|---|---|
| `results/predictions_gw{N}.csv` | `results/2025-26/gw{N}/predictions.csv` | Direct move |
| `results/squad_gw{N}.csv` + `results/xi_gw{N}.csv` | `results/2025-26/gw{N}/optimal_squad.csv` | Merge (see below) |
| `results/recommend_gw{N}.csv` | `results/2025-26/gw{N}/recommend.csv` | Direct move |
| `results/squad_recommend_gw{N}.csv` + `results/xi_recommend_gw{N}.csv` | `results/2025-26/gw{N}/recommended_squad.csv` | Merge (see below) |
| `results/snapshots/bootstrap_gw{N}.json` | `results/snapshots/2025-26/gw{N}/bootstrap.json` | Direct move |

**Merge logic (squad + xi → combined CSV):**
Existing squad files columns: `element, name, position, team, now_cost, xP, raw_xP`. XI files are a subset (starters only, same columns). For each player in squad: `is_starter = element in xi["element"].values`. Bench players: `bench_order` = rank among non-starters by xP descending (1–4). `is_captain` and `is_vice_captain` = null for GW30–33 (not retroactively available). Drop `raw_xP` (still present in `predictions.csv`). Keep `element`.

**Also:**
- Delete `results/signal_unresolved.csv`
- Do NOT create `actual_squad.csv` or `actual_transfers.csv` for GW30–33
- `accuracy_log.csv` stays at `results/accuracy_log.csv` — no action needed

### `config.py`
- Add `CURRENT_SEASON = "2025-26"`
- Add `def gw_dir(season: str, gw: int) -> Path: return RESULTS_DIR / season / f"gw{gw}"`
- Add `def snapshot_dir(season: str, gw: int) -> Path: return SNAPSHOTS_DIR / season / f"gw{gw}"`
- `SNAPSHOTS_DIR` remains `RESULTS_DIR / "snapshots"` (root unchanged — preserves `price_changes_latest.txt` at `SNAPSHOTS_DIR / "price_changes_latest.txt"`)
- Remove `SIGNAL_UNRESOLVED_CSV` constant

### `src/pipeline/datasources/signals.py`
`log_unresolved_name` is called from `ffs.py`, `premierinjuries.py`, and `reddit.py` — all pass `name`, `source`, `raw_text`, `timestamp` kwargs only (no `csv_path`). Change: keep function signature intact (removing it would break `__init__.py` export and three callers), but replace body with `logging.warning(f"[{source}] Unresolved player: {name!r} — {raw_text[:80]!r}")`. Remove the `from src.config import SIGNAL_UNRESOLVED_CSV` import inside the function body.

`__init__.py` export of `log_unresolved_name` stays (callers import from there).

### `src/pipeline/run.py`
- Add `_build_squad_csv(result: dict) -> pd.DataFrame` helper
- Add `_fetch_actual_transfers(entry_id: int, gw: int, bootstrap: dict) -> list[dict]` helper
- `phase_predict`: write `gw_dir(CURRENT_SEASON, gw) / "predictions.csv"` and `"optimal_squad.csv"` via `_build_squad_csv`; remove old `xi_{gw_label}.csv` + `squad_{gw_label}.csv` writes
- `phase_recommend`: write `gw_dir() / "recommend.csv"` and `"recommended_squad.csv"` via `_build_squad_csv`; remove old `squad_recommend_*` + `xi_recommend_*` writes
- Snapshot read/write: replace all `SNAPSHOTS_DIR / f"bootstrap_gw{N}.json"` with `snapshot_dir(CURRENT_SEASON, gw) / "bootstrap.json"`
- Fallback glob (currently `snapshot_dir.glob("bootstrap_gw*.json")`): replace with `sorted((SNAPSHOTS_DIR / CURRENT_SEASON).glob("*/bootstrap.json"), key=lambda p: int(p.parent.name.lstrip("gw")), reverse=True)`
- `phase_post_gw`: (1) write `actual_squad.csv` via `build_actual_squad_csv()` from `analysis.py`; (2) call `_fetch_actual_transfers()` and append to `results/actual_transfers.csv`; (3) call `subprocess.run(["python", "scripts/generate_reports.py", "--from-gw", "31"])`

### `src/pipeline/analysis.py`
- New function `build_actual_squad_csv(entry_picks: list[dict], bootstrap: dict, actual_pts_by_element: dict[int, int]) -> pd.DataFrame` — assembles `actual_squad.csv` schema from picks API response + bootstrap join + actual pts dict. Called from `run.py::phase_post_gw`.
- `append_accuracy_log` unchanged; writes to `RESULTS_DIR / "accuracy_log.csv"` (path stays at root)

### `scripts/generate_reports.py` (new)
```
python scripts/generate_reports.py [--from-gw 31]
```
- Reads `results/accuracy_log.csv`
- Reads `results/actual_transfers.csv` (may not exist for all GWs — handle gracefully)
- Reads `results/2025-26/gw{N}/recommend.csv` for each GW ≥ from_gw
- Writes `results/reports/rank_comparison.png`

### `.github/workflows/daily_bootstrap.yml`
- Snapshot write step: use `snapshots/2025-26/gw{N}/bootstrap.json` path
- Snapshot commit glob: `results/snapshots/2025-26/` instead of `results/snapshots/`
- After post-gw step, gated on `gw_finished == 'true'`: add step `python scripts/generate_reports.py --from-gw 31`
- Results commit glob: add `results/reports/rank_comparison.png`
- `price_changes_latest.txt` at `results/snapshots/price_changes_latest.txt` — path unchanged

---

## Migration Plan

1. Run `scripts/migrate_results.py` once locally
2. Verify `results/2025-26/gw{30..33}/` contents
3. Commit reorganised `results/` to git
4. Delete `scripts/migrate_results.py`, commit deletion
5. Pipeline writes new paths from GW34+ automatically

---

## Out of Scope

- Interactive web dashboard (Track F, next season)
- Historical `actual_squad.csv` backfill for GW30–33 (no retroactive actual_pts without re-fetching live match data)
- `raw_xP` dropped from squad CSVs (still present in `predictions.csv` for model evaluation)

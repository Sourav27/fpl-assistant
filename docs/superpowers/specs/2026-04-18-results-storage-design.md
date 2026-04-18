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
      price_changes_latest.txt  ← always current; stays at snapshots/ root
  accuracy_log.csv           ← season-level per-GW benchmark log (unchanged)
  actual_transfers.csv       ← season-level actual transfer history (new)
  reports/
    rank_comparison.png      ← regenerated each post-gw
```

**Key invariant:** `actual_squad.csv` is written only by `post-gw`. The `predict` and `recommend` phases write only `predictions.csv`, `optimal_squad.csv`, `recommend.csv`, and `recommended_squad.csv`. Re-running predict for model testing never touches `actual_squad.csv`.

---

## File Schemas

### `optimal_squad.csv` / `recommended_squad.csv`
Written by predict/recommend phase. Contains xP predictions only.

| Column | Type | Description |
|--------|------|-------------|
| name | str | Player display name |
| position | str | GK / DEF / MID / FWD |
| team | str | Club name |
| xP | float | Predicted points this GW |
| is_starter | bool | True = XI, False = bench |
| bench_order | int | 1–4 for bench players, null for starters |
| is_captain | bool | |
| is_vice_captain | bool | |
| now_cost | float | Price in £M |

### `actual_squad.csv`
Written by post-gw only. Never overwritten by predict.

| Column | Type | Description |
|--------|------|-------------|
| name | str | Player display name |
| position | str | GK / DEF / MID / FWD |
| team | str | Club name |
| actual_pts | int | Points scored this GW (incl. captain multiplier) |
| is_starter | bool | |
| bench_order | int | 1–4 for bench, null for starters |
| is_captain | bool | |
| is_vice_captain | bool | |
| now_cost | float | Price in £M |

### `actual_transfers.csv`
Season-level. One row per transfer made. Written/appended by post-gw.
Fetched from FPL API: `/api/entry/{entry_id}/transfers/`

| Column | Type | Description |
|--------|------|-------------|
| gw | int | Gameweek |
| player_out | str | Player transferred out |
| player_in | str | Player transferred in |
| transfer_rank | int | Order within GW (1-based) |
| actual_pts_gained | float | player_in actual_pts − player_out actual_pts that GW |
| hit_taken | bool | True if this transfer cost -4 pts |

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

## reports/rank_comparison.png

Regenerated each post-gw. Covers GW31 onwards.

**X-axis:** Gameweek number
**Y-axis:** Approximate rank percentile (lower = better; displayed as "top X%")

**Three lines:**
- **My team** — `your_pts` from `accuracy_log.csv`
- **Optimal squad** — `wildcard_pts` from `accuracy_log.csv`
- **Recommended squad** — `recommended_pts` from `accuracy_log.csv`

Rank percentile interpolated from benchmark thresholds already in `accuracy_log.csv` (`top_1k_score`, `top_10k_score`, `top_100k_score`, `avg_score`, `ranked_count`).

**Decision impact subplot** (below main chart): bar chart per GW showing `actual_pts_gained − recommended_pts_gained` from `actual_transfers.csv` vs `recommend.csv`. Green = followed recommendation and gained; red = diverged and lost.

---

## New & Changed Components

### `scripts/migrate_results.py` (one-off, deleted after use)
- Moves `results/predictions_gw{N}.csv` → `results/2025-26/gw{N}/predictions.csv`
- Renames `squad_gw{N}.csv` → `optimal_squad.csv`, `xi_gw{N}.csv` merged into it (XI + bench combined)
- Moves `recommend_gw{N}.csv` → `recommend.csv`
- Renames `squad_recommend_gw{N}.csv` → `recommended_squad.csv`, `xi_recommend_gw{N}.csv` merged
- Moves `results/snapshots/bootstrap_gw{N}.json` → `results/snapshots/2025-26/gw{N}/bootstrap.json`
- Removes `signal_unresolved.csv`
- Does NOT create `actual_squad.csv` or `actual_transfers.csv` for past GWs (no retroactive actual_pts)

### `config.py`
- Add `gw_dir(season: str, gw: int) -> Path` helper returning `RESULTS_DIR / season / f"gw{gw}"`
- Add `snapshot_dir(season: str, gw: int) -> Path` helper
- Remove inline path constructions for per-GW files across pipeline modules
- `CURRENT_SEASON = "2025-26"` constant

### `src/pipeline/run.py`
- All per-GW output paths use `gw_dir(CURRENT_SEASON, gw)`
- `phase_predict`: writes `predictions.csv`, `optimal_squad.csv` (combined XI+bench)
- `phase_recommend`: writes `recommend.csv`, `recommended_squad.csv` (combined XI+bench)
- `phase_post_gw`: writes `actual_squad.csv` to `gw_dir()`; fetches transfers from FPL API and appends to `actual_transfers.csv`; calls `generate_reports.py`

### `src/pipeline/analysis.py`
- Updated to write `actual_squad.csv` with actual_pts per player
- Populate `actual_pts_gained` in `actual_transfers.csv` by joining transfer history against `actual_squad.csv`

### `scripts/generate_reports.py` (new)
- Reads `accuracy_log.csv` + `actual_transfers.csv` + per-GW `recommend.csv` files
- Writes `results/reports/rank_comparison.png`
- CLI: `python scripts/generate_reports.py --from-gw 31`

### `.github/workflows/daily_bootstrap.yml`
- Update snapshot path to `snapshots/2025-26/gw{N}/bootstrap.json`
- Call `generate_reports.py` after post-gw step

---

## Migration Plan

1. Run `scripts/migrate_results.py` once on local machine
2. Commit reorganised `results/` to git
3. Delete `scripts/migrate_results.py`
4. Update pipeline to write new paths going forward (GW34+)

---

## Out of Scope

- Interactive web dashboard (Track F, next season)
- Historical `actual_squad.csv` backfill for GW30–33 (no retroactive actual_pts without re-fetching live data)
- `signal_unresolved.csv` — removed; unresolved signals logged to console only

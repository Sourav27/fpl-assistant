# Weekly Data Strategy Design

**Date:** 2026-03-27
**Status:** Approved
**Scope:** How the FPL weekly pipeline handles data accumulation, model retraining, vaastav/live data combination, and player availability filtering for GW32-38.

---

## Context

The FPL prediction pipeline runs weekly across three phases (pre-deadline, predict, post-gw). Three architectural decisions were needed that the original implementation plan did not fully address:

1. Whether and how prediction models are retrained as new data arrives
2. How historical vaastav data combines with live FPL API data for the current season
3. How player availability/injury information filters into team selection

### Constraints

- Only 7 gameweeks remain (GW32-38, Apr 10 - May 24, 2026)
- vaastav/Fantasy-Premier-League has 2025-26 data through GW29 as of today
- vaastav updates ~3 times per season (not weekly)
- FPL API endpoints require no authentication and are reliably available
- Future integration with X.com (Twitter) sources for availability intel is planned but out of scope

---

## Decision 1: Model Retraining Strategy

**Choice:** Static model with manual retraining (Option A)

### Design

- The trained model (`models/rf_model.sav`) remains frozen during weekly pipeline runs
- Each week, new GW results are appended to the merged dataset automatically
- The pipeline prepares training-ready data each week so retraining is a one-command operation
- Retraining is a manual decision by the user — triggered when prediction accuracy degrades or after significant new data accumulates

### Rationale

- 7 remaining gameweeks is too few for automated retraining to show meaningful improvement
- Automated retraining adds complexity and risk (a bad retrain could degrade predictions with no time to recover)
- Manual control lets the user inspect model performance before committing to a new model

### Implementation

- `run.py` gains a `retrain` phase that builds the full feature-engineered dataset and trains a new Random Forest model (RF only — positional models and XGBoost are out of scope for the 7-GW timeline)
- The new model is saved with a timestamped filename (e.g., `rf_model_gw33.sav`) alongside the existing one
- `config.py` gains `ACTIVE_MODEL = MODELS_DIR / "rf_model.sav"` — the user promotes a new model by updating this path
- The `retrain` phase prints comparison metrics (MAE, R2) between old and new model on held-out data to support the decision

---

## Decision 2: Data Combination — Vaastav Base + Live API Patch

**Choice:** Use vaastav as the historical base, patch forward with live FPL API data for missing gameweeks.

### Data Layers

| Layer | Source | Coverage | Update Cadence |
|-------|--------|----------|----------------|
| Historical seasons | vaastav `merged_gw.csv` | 2016-17 through 2024-25 | Static (complete) |
| Current season base | vaastav `merged_gw.csv` | 2025-26 GW1-29 | Updated ~3x/season |
| Current season live | FPL API `element-summary/{id}/` | 2025-26 GW30+ | After each GW |

### Schema Normalization

The FPL API player history fields do not match vaastav column names exactly. A normalization mapping is required:

| FPL API field | vaastav column | Notes |
|---------------|----------------|-------|
| `expected_goals` | `expected_goals` | Same in 2025-26; older seasons may differ |
| `expected_assists` | `expected_assists` | Same |
| `expected_goal_involvements` | `expected_goal_involvements` | Added 2024-25 |
| `expected_goals_conceded` | `expected_goals_conceded` | Added 2024-25 |
| `opponent_team` | `opponent_team` | Team ID (integer) |
| `total_points` | `total_points` | Same |
| `value` | `value` | Player cost in 0.1M units |

**Column categories:**

| Category | Columns | Handling in `_live.csv` |
|----------|---------|------------------------|
| Directly mapped | `total_points`, `minutes`, `goals_scored`, `assists`, `clean_sheets`, `bonus`, `bps`, `ict_index`, `influence`, `creativity`, `threat`, `value`, `transfers_in`, `transfers_out`, `selected`, `was_home`, `opponent_team`, `fixture`, `round`, `kickoff_time`, `expected_goals`, `expected_assists`, `expected_goal_involvements`, `expected_goals_conceded` | Copy directly from API |
| Derived post-fetch | `team` (name), `position` (string), `name`, `xP`, `GW`, `season` | Derived from bootstrap-static lookups |
| Unavailable from API | `clearances_blocks_interceptions`, `defensive_contribution`, `recoveries`, `tackles` | Fill with NaN — these are NOT used in the model's feature set |

Key differences to handle:
- API returns numeric team IDs; vaastav's `team` column uses team names — resolve via bootstrap `teams` lookup
- API's `fixture` is a fixture ID; needs join with fixtures endpoint for FDR
- The model's 18 feature columns (`total_points_roll_4`, `ict_index_roll_4`, `transfers_net`, etc.) are all derivable from the "directly mapped" columns above, so NaN in the "unavailable" columns does not affect predictions

### Player Identity

- The bootstrap-static endpoint's `elements` list is the authoritative current roster
- Any player not present in bootstrap (transferred out of PL, retired) is excluded from predictions regardless of vaastav history
- New January signings with no historical rows get xP from the API's `ep_this`/`ep_next` fields as a fallback (no rolling features available)
- Mid-season club transfers: the player's `team` updates in bootstrap; historical rows retain the old team (this is correct — features reflect where they played)

### API Failure Handling

- All API calls use exponential backoff (3 retries, 1s/2s/4s delays) — already in `fetch.py`
- If the API is unreachable during `post-gw`, skip live data collection; log a warning and fall back to last available dataset
- If the API is unreachable during `pre-deadline`, skip xP capture and use the model's predicted xP instead (no availability filtering that week)
- Bootstrap snapshots are cached as JSON in `results/snapshots/` — if a fresh fetch fails, use the most recent snapshot if < 48 hours old

### Rate Limiting

The `post-gw` phase fetches `element-summary/{id}/` per player. To avoid overwhelming the API:
- Fetch only players present in the current bootstrap (~500-600 active players, not all ~700)
- 0.5s sleep between requests (~5 minutes total runtime)
- Log progress every 50 players

### Data Flow

```
post-gw phase:
  1. Fetch bootstrap-static → get team ID-to-name mapping
  2. For each player: fetch element-summary/{id}/ → extract history[gw]
  3. Normalize API fields to vaastav schema
  4. Save as data/Fantasy-Premier-League/data/2025-26/gws/gw{N}_live.csv

prepare.py (build_merged_dataset):
  1. Load vaastav merged_gw.csv for all seasons
  2. Load any gw{N}_live.csv files for current season
  3. Deduplicate by (element, GW) — prefer vaastav over live (richer columns);
     only use _live.csv for GWs not covered in vaastav's merged_gw.csv
  4. Concatenate into unified dataset
```

### File Convention

- Live-patched GW files: `data/Fantasy-Premier-League/data/2025-26/gws/gw{N}_live.csv`
- The `_live` suffix distinguishes them from vaastav's `gw{N}.csv` files
- `merged_gw.csv` from vaastav is never modified

---

## Decision 3: Player Availability Filtering — Hybrid (Option C)

**Choice:** Hard-exclude near-certain absentees, soft-scale doubtful players' xP.

### FPL API Availability Fields

| Field | Values | Used |
|-------|--------|------|
| `status` | `a` (available), `d` (doubtful), `i` (injured), `u` (unavailable), `s` (suspended), `n` (not in squad) | Yes |
| `chance_of_playing_next_round` | `0`, `25`, `50`, `75`, `100`, `null` | Yes |
| `news` | Free text injury/suspension detail | Logged, not parsed |

### Filtering Rules (Decision Table)

Evaluated top-to-bottom; first matching rule wins:

| # | `status` | `chance_of_playing_next_round` | Action | Rationale |
|---|----------|-------------------------------|--------|-----------|
| 1 | `i`, `u`, `s`, `n` | any | **Hard exclude** | Confirmed out |
| 2 | any | `0` or `25` | **Hard exclude** | Near-certain to miss |
| 3 | any | `50` | **Soft scale: xP * 0.50** | Coin flip — halve expected value |
| 4 | `d` | `null` | **Soft scale: xP * 0.50** | Doubtful with no probability = treat as 50/50 |
| 5 | any | `75` | **Soft scale: xP * 0.75** | Likely to play but some risk |
| 6 | `a` | `100` or `null` | **No adjustment** | Available, no concerns |
| 7 | `d` | `100` | **No adjustment** | Flagged doubtful but FPL says 100% — trust the number |

### Implementation

- New function `filter_availability(predictions_df, bootstrap_data)` in the pipeline
- Called between `predict_next_gw()` and `optimize_team()` in the orchestrator
- Returns a filtered/adjusted DataFrame ready for the optimizer
- Logs excluded and scaled players for transparency

### Extensibility

The filtering layer is designed to be modular. Future signals (e.g., X.com reliability sources) can be added as additional inputs to `filter_availability()` without changing the optimizer or prediction modules.

---

## Impact on Existing Plan

These decisions require the following additions to the implementation plan at `docs/superpowers/plans/2026-03-27-fpl-weekly-pipeline.md`:

1. **Task 2 (fetch.py):** Add function to fetch and normalize individual player GW history to vaastav schema
2. **Task 3 (prepare.py):** Add `_live.csv` discovery and deduplication logic to `build_merged_dataset()`
3. **Task 5 (predict.py) or new module:** Add `filter_availability()` function
4. **Task 7 (run.py):**
   - `post-gw` phase: collect player histories and save as `gw{N}_live.csv`
   - `predict` phase: call `filter_availability()` between prediction and optimization
   - New `retrain` phase: build full dataset and train new model
5. **New tests:** `test_filter_availability()`, `test_live_gw_normalization()`, `test_deduplication()`

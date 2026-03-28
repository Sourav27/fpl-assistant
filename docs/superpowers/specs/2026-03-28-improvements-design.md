# FPL Pipeline Improvements — Design Spec

**Date:** 2026-03-28
**Status:** Approved
**Depends on:** Weekly pipeline (src/pipeline/) being functional with predict phase working

---

## Overview

Six work items across five priorities that evolve the FPL pipeline from a standalone
prediction engine into a personalised weekly assistant. Two independent tracks:

- **Track A (user-facing):** P1 → P2a — team sync, transfer recommendations, post-match analysis
- **Track B (model quality):** P4 → P3 → P5 — feature improvements, positional models, fallback benchmarking

Tracks can be parallelised.

---

## P1 — User Team Sync + Recommend Phase

### Problem

The current optimizer builds a squad from scratch with a fixed £100M budget. It doesn't
know the user's actual squad, bank balance, free transfers, or selling prices. This makes
its output aspirational rather than actionable.

### Solution

New `recommend` CLI phase that fetches the user's team state from the FPL API and runs
a transfer-aware multi-GW optimizer.

### User Config

**File:** `user_config.yaml` (gitignored) + `user_config.example.yaml` (committed)

```yaml
# user_config.example.yaml
teams:
  default:
    entry_id: 1234567
    label: "Main"
  alt:
    entry_id: 7654321
    label: "Experimental"

preferences:
  horizon_gws: 5           # GWs to plan ahead (1 = single-GW, max 5)
  max_hit_points: 8         # won't recommend more than -8 hits in a single GW
```

**Loading:** New `load_user_config()` function reads the YAML. If file is missing,
print a helpful error pointing at the example file.

### New Module: `src/pipeline/user.py`

Fetches user team state from the public FPL API (no auth required):

| Endpoint | Data |
|----------|------|
| `/api/entry/{id}/` | Overall info, bank balance |
| `/api/entry/{id}/event/{gw}/picks/` | Current 15-player squad, captain, bench order |
| `/api/entry/{id}/transfers/` | Transfer history (needed to compute selling prices) |
| `/api/entry/{id}/history/` | GW-by-GW points, banked free transfers |

Returns a dataclass:

```python
@dataclass
class UserTeamState:
    entry_id: int
    current_squad: list[int]       # 15 element IDs
    selling_prices: dict[int, int] # element → selling price (0.1M units)
    bank: int                      # remaining budget (0.1M units)
    free_transfers: int            # banked free transfers (1-5)
    active_chip: str | None        # "wildcard", "freehit", "bboost", "3xc", or None
    total_value: int               # sum of selling prices + bank
```

**Selling price computation:** FPL API provides `selling_price` directly in the picks
endpoint for the user's players. If unavailable, compute as:
`purchase_price + floor((current_price - purchase_price) / 2)` rounded to nearest 0.1M.

### New Module: `src/pipeline/recommend.py`

Two modes controlled by a single `horizon` parameter:

#### Transfer Mode (horizon ≥ 2, default 5)

Multi-GW ILP formulation using PuLP:

**Decision variables:**
- `squad[player][gw]` ∈ {0, 1} — player in squad at each GW
- `xi[player][gw]` ∈ {0, 1} — player in starting XI
- `transfer_in[player][gw]` ∈ {0, 1}
- `transfer_out[player][gw]` ∈ {0, 1}
- `captain[player][gw]` ∈ {0, 1}

**Objective:** Maximise:
```
sum over gw: sum over players: xP[player][gw] * fdr_weight[gw] * (1 + captain[player][gw])
  - 4 * max(0, transfers_used[gw] - free_transfers[gw])
```

Where `xP[player][gw]` uses the current prediction adjusted by fixture difficulty
rating for future GWs.

**Constraints per GW:**
- Squad validity: 2 GK, 5 DEF, 5 MID, 3 FWD
- XI validity: 1 GK, ≥3 DEF, ≥2 MID, ≥1 FWD, total = 11
- Max 3 per club
- Budget: sum(cost of squad) ≤ total_value at each GW
- Transfer continuity: `squad[gw] = squad[gw-1] + transfers_in[gw] - transfers_out[gw]`
- Free transfer tracking: carries forward unused (max 5), resets to 1 after use
- `max_hit_points` cap per GW

**FDR weighting for future GWs:**
Use fixture difficulty ratings from the FPL API fixtures endpoint. For each player,
look up their team's opponent difficulty in each future GW. Apply a simple inverse
scaling: `fdr_weight = (6 - fdr) / 4` so FDR 2 (easy) → 1.0, FDR 5 (hard) → 0.25.
GW 1 (current) uses raw xP with no discount.

**Output:**
```
GW33: Transfer OUT Watkins (5.2 xP) → IN Haaland (7.8 xP)  [1 free transfer]
GW34: Hold
GW35: Transfer OUT Saka (6.1 xP) → IN Palmer (7.0 xP)  [1 free transfer]
Projected total xP (5 GWs): 312.4
Transfer cost: 0 points (all free transfers)
```

Saved to `results/recommend_gw{N}.csv` with columns:
`gw, action, player_out, player_in, xP_out, xP_in, transfer_cost, squad_after`

#### Wildcard Mode (`--wildcard` flag)

- Ignores current squad and transfer constraints
- Budget = `team_state.total_value`
- Single-GW optimisation (same as current `optimize_team()` but with real budget)
- Also auto-activates when API reports wildcard/free-hit chip is active

#### Horizon = 1 (no flag)

- Respects current squad and free transfers
- Recommends best transfers for this GW only
- Useful for quick "what should I do right now?" advice

### CLI Interface

```bash
# 5-GW transfer plan (default)
python -m src.pipeline.run recommend --gw 33

# Single-GW transfers
python -m src.pipeline.run recommend --gw 33 --horizon 1

# Wildcard / planning mode
python -m src.pipeline.run recommend --gw 33 --wildcard

# Alternate team
python -m src.pipeline.run recommend --gw 33 --team alt
```

**Prerequisite:** `predict` must have been run first for the target GW (reads from
`results/xi_gw{N}.csv` or equivalent predictions output). The `recommend` phase does
NOT re-run predictions.

### Chip Strategy

Deferred to future work. The current design only handles wildcard auto-detection.
Future extension: recommend optimal chip timing (bench boost on DGW, triple captain
on high-xP fixture, free hit during blank GW).

---

## P2a — Post-Match Predicted vs Actual Analysis

### Problem

No feedback loop exists. After a GW, there's no way to see where predictions were
wrong or how the recommended team compared to what actually happened.

### Solution

Extend `phase_post_gw()` to run a three-way comparison after collecting live data.

### Three Comparisons

1. **Your actual team vs predicted xP**
   - For each player in your GW picks, show predicted xP vs actual points
   - Highlight biggest misses (positive and negative)

2. **Recommended team vs actual points**
   - Load the `recommend` output for this GW (if it exists)
   - Compare recommended XI actual points vs your actual XI points
   - Shows "value left on the table"

3. **GW dream team vs recommended**
   - Fetch dream team from FPL API (`/api/dream-team/{gw}/` or from bootstrap event data)
   - Compare recommended XI vs dream team — how close to the ceiling?

### Output

**Terminal summary:**
```
=== GW33 Post-Match Analysis ===

Your Team:  58 pts (predicted: 72.3 xP)
Recommended: 65 pts (predicted: 78.1 xP)
Dream Team:  89 pts

Biggest prediction misses (your team):
  Haaland:  predicted 8.5 xP, actual 2 pts  (-6.5)
  Palmer:   predicted 4.2 xP, actual 12 pts (+7.8)

Recommendation value: +7 pts over your team this GW
Dream team gap: -24 pts (recommended vs ceiling)
```

**Season log:** Append one row per GW to `results/accuracy_log.csv`:
```csv
gw, your_pts, your_predicted_xp, recommended_pts, recommended_xp, dream_team_pts, timestamp
33, 58, 72.3, 65, 78.1, 89, 2026-04-12T20:00:00Z
```

### Data Dependencies

- User team picks: fetched via `user.py` (P1 must be implemented first)
- Recommended team: loaded from `results/recommend_gw{N}.csv` (optional — skip comparison if missing)
- Dream team: fetched from FPL API
- Predictions: loaded from existing `results/xi_gw{N}.csv`
- Actual points: from live GW data (already collected by `phase_post_gw()`)

### CLI

No new flags — this runs automatically as part of `post-gw`:
```bash
python -m src.pipeline.run post-gw
```

If `user_config.yaml` is missing, skip user-team comparison with a warning.

---

## P4 — Leading Indicator Features + Understat Revival (Research Spike)

### Problem

The model's top feature is `minutes` (a lagging indicator). Leading indicators like
xG, xA, and xMin from Understat are commented out because the scraper is broken
(hardcoded to season 2024). The current feature set over-weights what happened last
week rather than what's likely to happen next week.

### Goal

Revive the Understat scraper, add xG/xA/xMin features to the pipeline, and measure
whether they improve prediction accuracy (MAE reduction).

### Approach

1. **Fix `_original/data_collection/understat.py`** — update season parameter to be
   dynamic, test against current Understat DOM structure
2. **Add xG/xA join to `prepare.build_merged_dataset()`** — match Understat player
   names to FPL player codes (fuzzy matching likely needed)
3. **Add rolling xG/xA features to `features.engineer_features()`** — `xG_roll_4`,
   `xA_roll_4`, `xMin_roll_4`
4. **Retrain model with new features, compare MAE** — if MAE improves, adopt;
   if not, keep features available but don't default to them
5. **Evaluate feature importance ranking** — verify that leading indicators rank
   higher than lagging ones in the new model

### Deliverables

- Working Understat scraper in `src/pipeline/understat.py` (new location, not restored to old path)
- Feature comparison report: old features vs new features MAE/R²
- Updated `get_feature_columns()` if new features adopted

### Risk

Medium. Understat DOM may have changed. May need `playwright` or `httpx` with JS
rendering instead of `requests` + `BeautifulSoup`. FPL-to-Understat player name
matching is imperfect (fuzzy match on web_name).

---

## P3 — Positional Prediction Models (Research Spike, depends on P4)

### Problem

GK scoring (clean sheets, saves) is fundamentally different from FWD scoring (goals,
assists). A single global model may miss position-specific patterns. NB05 showed
GK MAE 0.770 vs FWD MAE 1.249 but used the same features for all positions.

### Goal

Test whether position-specific models with position-specific features outperform the
global model. Only adopt if measurably better.

### Approach

1. **Define position-specific feature sets:**
   - GK: `clean_sheets_roll_4`, `saves_roll_4`, `goals_conceded_roll_4`, `xGC_roll_4`
   - DEF: `clean_sheets_roll_4`, `tackles_roll_4`, `clearances_roll_4`, `xGC_roll_4`
   - MID: `xG_roll_4`, `xA_roll_4`, `creativity_roll_4`, `key_passes_roll_4`
   - FWD: `xG_roll_4`, `shots_roll_4`, `npxG_roll_4`, `big_chances_roll_4`

2. **Train 4 models** (one per position) using XGBoost (from P4/improvement #1)

3. **Compare per-position MAE** vs global model on holdout set

4. **If better:** Update `predict_next_gw()` to route players to position-specific models
   **If worse:** Document findings, keep global model

### Deliverables

- Comparison table: global MAE vs positional MAE per position
- If adopted: 4 model files + updated predict.py routing logic
- If rejected: research findings document explaining why

### Dependency

Requires P4 (Understat features) to be meaningful — without xG/xA, the
position-specific feature sets won't be different enough from the global set.

---

## P5 — Fallback Benchmarking (Research Spike)

### Problem

When the trained model is missing or stale, the pipeline falls back to `ep_this` from
the FPL bootstrap API. This may not be the best available fallback.

### Goal

Benchmark alternative fallback strategies and pick the best one by MAE.

### Candidates to Benchmark

1. **`ep_this`** (current) — FPL's own expected points for current GW
2. **`ep_next`** — FPL's expected points for the next GW
3. **Weighted rolling xP** — `0.5 * xP_roll_4 + 0.3 * xP_roll_8 + 0.2 * season_avg`
4. **Bootstrap composite** — lightweight regression on `form`, `selected_by_percent`,
   `ict_index`, `ep_next` (all available without a trained model)
5. **Naive last-3 average** — mean of last 3 GW actual points

### Approach

1. For each candidate, compute predicted xP for all players across 5+ historical GWs
2. Compare MAE against actual points
3. Pick the lowest-MAE fallback as the new default
4. Implement in `run.py`'s `_fallback` code path

### Deliverables

- Benchmark results table (MAE per candidate per GW)
- Updated fallback logic in `run.py` using the winner
- If `ep_next` is best, just swap the field name (trivial change)

---

## Implementation Order

```
Track A (user-facing):              Track B (model quality):
  P1: User Team Sync + Recommend      P4: Understat Revival + Features
      ↓                                    ↓
  P2a: Post-Match Analysis             P3: Positional Models
                                           ↓
                                       P5: Fallback Benchmarking
```

Tracks are independent. Within each track, order is sequential (each item depends
on the previous).

### Suggested Model Assignment for Subagents

| Task | Complexity | Suggested Model |
|------|-----------|-----------------|
| P1: user_config.yaml + example | Mechanical | haiku |
| P1: src/pipeline/user.py (API fetch + dataclass) | Mechanical | haiku |
| P1: src/pipeline/recommend.py (multi-GW ILP) | High — complex ILP formulation | opus |
| P1: CLI integration in run.py | Standard | sonnet |
| P1: Tests for user.py + recommend.py | Standard | sonnet |
| P2a: Post-match analysis extension | Standard | sonnet |
| P2a: Dream team fetch + comparison | Mechanical | haiku |
| P2a: accuracy_log.csv accumulation | Mechanical | haiku |
| P4: Understat scraper revival | Standard — web scraping | sonnet |
| P4: Feature integration + retrain comparison | Standard | sonnet |
| P3: Positional model research | High — design judgment | opus |
| P5: Fallback benchmarking | Standard | sonnet |

### File Structure (New/Modified)

```
New files:
  user_config.yaml              # gitignored — user's team IDs + preferences
  user_config.example.yaml      # committed — reference template
  src/pipeline/user.py          # FPL API user data fetcher
  src/pipeline/recommend.py     # transfer-aware multi-GW optimizer
  src/pipeline/understat.py     # revived Understat scraper (P4)
  results/recommend_gw{N}.csv   # transfer plan output
  results/accuracy_log.csv      # season-long prediction accuracy

Modified files:
  src/pipeline/run.py           # add recommend phase, extend post-gw
  src/pipeline/features.py      # add xG/xA rolling features (P4)
  src/pipeline/prepare.py       # join Understat data (P4)
  src/pipeline/predict.py       # positional model routing (P3, if adopted)
  src/config.py                 # user config loading, new API endpoints
  .gitignore                    # add user_config.yaml
  docs/improvements-roadmap.md  # update status as items are completed
```

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
  fdr_sensitivity: 0.15     # how much fixture difficulty affects xP (0 = ignore, 0.3 = aggressive)
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
    current_squad: list[int]       # 15 element IDs (seasonal)
    squad_codes: list[int]         # 15 persistent player codes (for joining with predictions)
    selling_prices: dict[int, int] # element → selling price (0.1M units, e.g., 77 = £7.7m)
    bank: int                      # remaining budget (0.1M units, e.g., 350 = £35.0m)
    free_transfers: int            # banked free transfers (1-5)
    active_chip: str | None        # "wildcard", "freehit", "bboost", "3xc", or None
    total_value: int               # sum of selling prices + bank (0.1M units)
```

**Units note:** All cost values are stored in **0.1M unit increments** (same as FPL API: 10 = £1m, 77 = £7.7m).
For user-facing output (CSVs, logs, UI), divide by 10 and format as £ currency (e.g., 77 → £7.7m).

**Element ID vs code:** The FPL API returns `element` IDs (seasonal). The existing
pipeline uses persistent `code` as the cross-season identifier. `user.py` must map
element IDs to codes via the bootstrap lookup (`bootstrap["elements"]` contains both
`id` and `code` per player). The `recommend.py` module joins on `code` internally
but outputs `element` IDs for the user-facing transfer plan (since FPL uses element IDs
in the UI).

**Selling price computation:** The FPL picks endpoint (`/api/entry/{id}/event/{gw}/picks/`)
may include a `selling_price` field per pick, but this is not reliably available for
public (non-authenticated) access. **Primary path:** compute selling price from
transfer history using 0.1M units:
```
selling_price = purchase_price + floor((current_price - purchase_price) / 2)
```
Example: purchase 75 (£7.5m), current 78 (£7.8m) → selling 76 (£7.6m).
The transfer history endpoint (`/api/entry/{id}/transfers/`) provides `element_in_cost`
for each transfer, giving the purchase price in 0.1M units. For players in the initial
squad (no transfer record), use the player's starting season price from bootstrap.

**Config validation:** `load_user_config()` validates required keys (`teams.default.entry_id`),
types (entry_id must be int, horizon_gws 1-5), and prints specific errors for missing
or malformed fields. Uses simple dict checks — no extra dependencies.

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

**Additional decision variables:**
- `hits[gw]` ≥ 0 (integer) — extra transfers beyond free allowance
- `ft[gw]` ∈ {1..5} (integer) — banked free transfers entering this GW
- `used_ft[gw]` ∈ {0, 1} — whether any transfers were made this GW

**Objective:** Maximise:
```
sum over gw:
  sum over players: xP[player][gw] * fdr_weight[player][gw] * (1 + captain[player][gw])
  - 4 * hits[gw]
```

**Linearisation of transfer cost:** PuLP requires linear expressions. The `max(0, ...)`
is replaced by the auxiliary variable `hits[gw]` with constraints:
```
hits[gw] >= transfers_used[gw] - ft[gw]      (hits absorb excess)
hits[gw] >= 0                                 (no negative hits)
hits[gw] <= max_hit_points / 4                (user preference cap)
```

Where `xP[player][gw]` uses the current prediction adjusted by fixture difficulty
rating for future GWs.

**Constraints per GW:**
- Squad validity: 2 GK, 5 DEF, 5 MID, 3 FWD
- XI validity: 1 GK, ≥3 DEF, ≥2 MID, ≥1 FWD, total = 11
- Max 3 per club
- Transfer continuity: `squad[gw] = squad[gw-1] + transfers_in[gw] - transfers_out[gw]`
- `transfers_used[gw] = sum(transfer_in[player][gw])` for all players
- `max_hit_points` cap per GW: `hits[gw] * 4 <= max_hit_points`

**Budget constraint (asymmetric buy/sell):**
The budget must account for the selling price haircut. Players are sold at their
selling price (purchase + 50% profit rounded down), but bought at current market price.
All values in 0.1M units (same as FPL API: budget 1000 = £100M):
```
bank[gw] = bank[gw-1]
  + sum(selling_price[p] * transfer_out[p][gw])   # revenue from sales (0.1M units)
  - sum(now_cost[p] * transfer_in[p][gw])          # cost of purchases (0.1M units)
bank[gw] >= 0                                       # cannot go negative
```
Where `bank[0] = team_state.bank` and selling prices come from `UserTeamState` (computed in 0.1M units).
For user-facing display, convert: `bank_pounds = bank / 10` (e.g., 350 → £35.0m).
For newly acquired players (bought in GW k, sold in GW k+n), selling price equals
purchase price (no profit yet) — a simplification acceptable for a 5-GW horizon.

**Free transfer tracking (linearised with big-M):**
Free transfers carry forward when unused, capped at 5. This is conditional logic
that requires linearisation:
```
used_ft[gw] ∈ {0, 1}
transfers_used[gw] <= M * used_ft[gw]               (if transfers=0 then used_ft=0)
transfers_used[gw] >= used_ft[gw]                    (if transfers>0 then used_ft=1)
ft[gw+1] <= ft[gw] + 1 + M * used_ft[gw]            (upper bound: grow by at most 1 when unused)
ft[gw+1] <= 5                                        (hard cap — prevents infeasibility at ft=5)
ft[gw+1] >= 1                                        (minimum 1 after using transfers)
ft[gw+1] <= 1 + M * (1 - used_ft[gw])               (reset to 1 if used)
ft[0] = team_state.free_transfers                    (initial state from API)
```
Where `M = 20` (maximum transfers per GW per FPL rules).

Note: no lower-bound on `ft[gw+1]` when unused — the solver naturally maximises `ft`
(more free transfers = fewer future hits in the objective). This avoids infeasibility
when `ft[gw] = 5` where a lower bound would force `ft[gw+1] = 6 > 5`.

**Blank and Double Gameweeks:**
Players with no fixture in a future GW (blank GW) get `xP = 0` for that week.
Players with two fixtures (double GW) get `xP` summed across both fixtures.
Fixture data from the FPL API fixtures endpoint determines BGW/DGW status per team.

**FDR weighting for future GWs:**
Fixture Difficulty Rating (FDR) measures how hard a given fixture is **for a specific team**
on a 1-5 scale (per FPL FAQ). Each fixture has two separate FDR values — one per team:
- `team_h_difficulty`: how hard the fixture is **for the home team**
- `team_a_difficulty`: how hard the fixture is **for the away team**

For xP weighting we want the player's own team FDR (not the opponent's). A player on
the home team uses `team_h_difficulty`; a player on the away team uses `team_a_difficulty`.
The existing pipeline names this `fdr_team` (see `prepare.py:add_fixture_difficulty()`).

For future GWs in the horizon, look up `fdr_team` from the FPL API fixtures endpoint:
```
fdr = team_h_difficulty  (if player's team is home)
fdr = team_a_difficulty  (if player's team is away)
```

Apply a configurable scaling function. Formula: `fdr_weight = 1.0 - fdr_sensitivity * (fdr - 3) / 2`

Default `fdr_sensitivity = 0.15` (configurable in `user_config.yaml`), giving:
- FDR 1 (very easy opponent) → 1.15 — boost xP
- FDR 2 (easy opponent) → 1.075
- FDR 3 (average) → 1.0
- FDR 4 (hard opponent) → 0.925
- FDR 5 (very hard opponent) → 0.85 — discount xP

This keeps weights in a reasonable range (0.85–1.15) and avoids extreme overweighting.
Current GW (GW 1 of the horizon) uses raw xP with no FDR adjustment.

**FDR data sourcing:**
FDR is embedded in the FPL API fixtures endpoint (`/api/fixtures/`) and the vaastav
`fixtures.csv`. The pipeline (`prepare.py:add_fixture_difficulty()`) joins FDR onto
historical GW data, creating `fdr_team` (difficulty for the player's own team) and
`fdr_opp` (difficulty for the opponent). For `recommend.py`, only `fdr_team` is used:
- Fetch fixtures from FPL API: `/api/fixtures/?event={gw}`
- For each player and future GW, find their team's fixture
- Extract `team_h_difficulty` if home, `team_a_difficulty` if away → this is `fdr_team`
- Apply the weighting formula above

If fixture data is unavailable, fall back to no FDR weighting (`fdr_weight = 1.0`).

**Player pool pre-filtering (solve time):**
A 5-GW horizon with 500+ players creates a large ILP. Pre-filter to top N players
per position by average xP across the horizon (e.g., top 30 DEF, top 20 MID, top 15 FWD,
top 10 GK) plus all players currently in the user's squad. All `now_cost` values from
predictions CSV are already in 0.1M units (from FPL API). Target solve time: < 60 seconds
on a consumer laptop. If solve time exceeds this, reduce N or horizon.

**Output (user-facing, all prices in £):**
```
GW33: Transfer OUT Watkins (5.2 xP, £5.2m) → IN Haaland (7.8 xP, £7.8m)  [1 free transfer]
GW34: Hold
GW35: Transfer OUT Saka (6.1 xP, £6.1m) → IN Palmer (7.0 xP, £7.0m)  [1 free transfer]
Projected total xP (5 GWs): 312.4
Transfer cost: 0 points (all free transfers)
Bank after transfers: £35.2m
```

Saved to `results/recommend_gw{N}.csv` with columns:
`gw, action, player_out, player_in, price_out, price_in, xp_out, xp_in, transfer_cost, bank_after`
(All prices in £ format for user readability, e.g., 5.2 for £5.2m)

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

**Prerequisite:** `predict` must have been run first for the target GW. The `predict`
phase must be updated to also save the full player predictions (not just the optimized XI)
to `results/predictions_gw{N}.csv` — all players with columns: `element, code, name,
position, team, xP, now_cost`. The `recommend` phase reads this file and does NOT re-run
predictions.

### Chip Strategy

Deferred to future work. The current design handles:
- **Wildcard/Free Hit:** auto-detected from API → switches to unconstrained mode
- **Bench Boost / Triple Captain:** if detected as active, log a warning
  ("bench boost active — recommend phase does not yet optimize bench selection")
  and proceed with normal optimization. Future extension will optimize bench for BB
  and identify TC targets.

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
   - Derive dream team from `/api/event/{gw}/live/` endpoint (highest-scoring XI by
     position rules) or from bootstrap `events[gw].top_element_info` if available.
     There is no dedicated `/api/dream-team/` endpoint — the dream team must be
     computed from live player scores.
   - Compare recommended XI vs dream team — how close to the ceiling?

### Output

**Terminal summary:**
```
=== GW33 Post-Match Analysis ===

Your Team:    58 pts  (predicted: 72.3 xP)  | Percentile rank: 20th
Recommended:  65 pts  (predicted: 78.1 xP)
Dream Team:   89 pts

Benchmark scores this GW:
  Best score:    109 pts  (rank 1)
  Top 1k:         85 pts
  Top 10k:        79 pts
  Top 100k:       73 pts
  Top 1M:         62 pts
  Average (50th): 38 pts

Biggest prediction misses (your team):
  Haaland:  predicted 8.5 xP, actual 2 pts  (-6.5)
  Palmer:   predicted 4.2 xP, actual 12 pts (+7.8)

Recommendation value: +7 pts over your team this GW
Dream team gap: -24 pts (recommended vs ceiling)
```

**Season log:** Append one row per GW to `results/accuracy_log.csv`:
```csv
gw, your_pts, your_predicted_xp, recommended_pts, recommended_xp, dream_team_pts,
    your_percentile_rank, best_score, top_1k_score, top_10k_score, top_100k_score,
    top_1m_score, avg_score, median_score, ranked_count, timestamp
33, 58, 72.3, 65, 78.1, 89, 20, 109, 85, 79, 73, 62, 38, 36, 12914049, 2026-04-12T20:00:00Z
```

### Benchmark Data Sources

Each benchmark maps to a specific FPL API source:

| Benchmark | Source | API Endpoint | Cost |
|-----------|--------|-------------|------|
| `best_score` | `events[gw].highest_score` | `bootstrap-static` | Free |
| `avg_score` | `events[gw].average_entry_score` | `bootstrap-static` | Free — **mean**, not median |
| `ranked_count` | `events[gw].ranked_count` | `bootstrap-static` | Free |
| `your_percentile_rank` | `current[gw].percentile_rank` | `entry/{id}/history/` | Free |
| `top_1k_score` | score at rank 1000 | Overall standings page 20 | 1 API call |
| `top_10k_score` | score at rank 10000 | Overall standings page 200 | 1 API call |
| `top_100k_score` | score at rank 100000 | Overall standings page 2000 | 1 API call |
| `top_1m_score` | score at rank 1000000 | Overall standings page 20000 | 1 API call (slow) |
| `median_score` | score at rank `ranked_count / 2` | Overall standings page `ranked_count / 100` | 1 API call (very slow) |

**`avg_score` vs `median_score`:** `avg_score` is the arithmetic mean from bootstrap (free, fast).
`median_score` is the true 50th percentile score — the score that exactly half of managers beat.
For a right-skewed score distribution (most managers cluster below the mean), median < mean.
Both are worth tracking: `avg_score` reflects total points in the population; `median_score`
is the more meaningful "did I beat half the field?" benchmark.

**`percentile_rank` values** are integers from the FPL-defined set
`[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]`,
where lower = better (top 1% = percentile_rank 1, median = percentile_rank 50).

**Overall standings pagination:** Each page returns 50 entries.
Endpoint: `/api/leagues-classic/{overall_league_id}/standings/?page_standings={page}`
The overall league ID is season-specific; look it up once from `/api/entry/{id}/leagues/`
— it is listed as the "Overall" classic league with ~13M members.

**Slow fetches (`top_1m_score`, `median_score`):** Pages 20000+ may time out.
Fetch with a 30s timeout; if it fails, write `null` and log a warning. These are
the lowest-priority benchmarks — `best_score`, `top_1k_score`, and `avg_score`
are most actionable.

### Data Dependencies

- User team picks: fetched via `user.py` (P1 must be implemented first)
- Recommended team: loaded from `results/recommend_gw{N}.csv` (optional — skip comparison if missing)
- Dream team: fetched from FPL API `/api/event/{gw}/live/`
- Predictions: loaded from `results/predictions_gw{N}.csv` (full player predictions)
- Actual points: from live GW data (already collected by `phase_post_gw()`)
- Benchmarks: bootstrap-static + overall league standings (4 targeted page fetches)

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

Recommended after P4 (Understat features) for best results — xG/xA make the
position-specific feature sets more distinct. However, a first-pass comparison can
use existing features (`clean_sheets_roll_4`, `creativity_roll_4`, `threat_roll_4`)
that are already in the pipeline. If P4 is delayed due to web-scraping risk, P3 can
proceed with current features as a proof-of-concept.

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
      ↓                                  ↓ (recommended, not strict)
  P2a: Post-Match Analysis             P3: Positional Models
                                           ↓
                                       P5: Fallback Benchmarking
```

Tracks are independent. Within Track A, P2a depends on P1 (needs user.py).
Within Track B, P3 benefits from P4 but can start with existing features as a
proof-of-concept if P4 is delayed. P5 is independent of P3/P4.

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
  results/predictions_gw{N}.csv  # full player predictions (all players, xP + cost)
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

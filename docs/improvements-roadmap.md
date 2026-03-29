# FPL Assistant — Improvements Roadmap

Distilled from archived research code (`_original/notebooks/`, `_original/optimization/`,
`_original/data_collection/`) before those folders were removed from version control.
Last updated: 2026-03-29.

---

## Baseline Performance (R Optimizer, 2022-23 Season)

The R-based `FPL_xPMin` script (`_original/optimization/FPL_xPMin.R`) ran over 33/38 GWs
and is the closest thing to a production benchmark:

| Metric | Value |
|--------|-------|
| GWs covered | 33 / 38 (GW 5–38, skipping blank GW 7) |
| Avg predicted xP | 98.9 pts/GW |
| Avg actual pts | 95.0 pts/GW |
| Prediction accuracy | 96% |
| Best GW | GW 29 — 148.1 xP predicted, 146 actual (Watkins captain) |
| Most-captained | Haaland (15/33 GWs) |

**What this tells us:** The optimizer already captures ~96% of achievable value.
Further accuracy gains must come from better *point prediction* (the ML model),
not from solver tuning.

---

## ML Model Baselines (Notebooks 04 & 05, 2022-23 data)

| Model | MAE | RMSE | R² | Notes |
|-------|-----|------|----|-------|
| Mean baseline | 1.556 | 2.372 | — | Predict every player scores the mean |
| Linear regression | 1.075 | 1.967 | — | Simple but competitive |
| Random Forest (global) | 1.035 | 1.948 | 0.313 | Current production model |
| XGBoost (global) | 1.026 | 1.952 | — | ~1% edge over RF; not yet in pipeline |
| RF positional — GK | 0.770 | — | 0.438 | Most predictable position |
| RF positional — DEF | 0.910 | — | — | |
| RF positional — MID | 1.048 | — | — | |
| RF positional — FWD | 1.249 | — | 0.321 | Least predictable |

**Key insight:** Global model R² ~0.31 means we explain ~31% of per-GW variance —
there is inherent randomness in football (33-GW observation confirms this).
Position-specific models do **not** consistently beat the global model; position is
already encoded as a feature in the global model.

---

## Top Predictive Features (NB04 feature importance)

Ranked by Random Forest mean decrease in impurity:

1. `minutes` / `minutes_roll_4` — 16–30% importance. **Playing time is king.** A
   two-stage model (predict P(starting) first, then conditional points) would likely
   improve R² by 3–5 pp.
2. `ict_index_roll_4` — influence + creativity + threat composite. Already in pipeline.
3. `total_points_roll_4` — form momentum. Already in pipeline.
4. `bps_roll_4` — bonus point system predicts bonus allocation.
5. `xG` / `xA` — **commented out in NB04** because Understat scraper was broken.
   These features drove a measurable improvement when the data was available.

---

## Implemented Features (Track A — 2026-03-29)

These are **new pipeline features** added in Track A (not the P1/P2/P3 improvement items below,
which are ML model quality improvements). Track A added the user-facing workflow layer on top
of the existing predict → optimize pipeline.

### Track A — User Team Sync (`src/pipeline/user.py`)

| Function | What it does |
|----------|-------------|
| `UserTeamState` | Dataclass holding squad (15 element IDs), squad codes, selling prices, bank, FTs, active chip. All costs in 0.1M units (FPL convention). |
| `fetch_user_team_state(entry_id, gw, bootstrap)` | Hits `/entry/{id}/event/{gw}/picks/`, `/entry/{id}/`, and `/entry/{id}/history/` to build a `UserTeamState`. Maps element IDs → persistent `code` values via bootstrap. |
| `compute_selling_price(purchase, current)` | FPL sell-price formula: `purchase + floor((current − purchase) / 2)`. No haircut on price drops. |
| `_compute_free_transfers(gw_history, current_gw)` | Simulates FT banking from match history. Unused FT banks 1 (cap 5); after transfers used, resets to 1 + 1. |
| `fetch_gw_benchmarks(gw, bootstrap, league_id)` | Reads `highest_score` / `average_entry_score` from bootstrap events; paginates the overall standings API to find top-1k / 10k / 100k / 1M cutoffs. |

### Track A — Transfer Recommender (`src/pipeline/recommend.py`)

| Function | What it does |
|----------|-------------|
| `compute_fdr_weight(fdr, sensitivity)` | `1.0 − sensitivity × (fdr_team − 3) / 2`. FDR 1 → boost, FDR 5 → discount. Clamped ≥ 0. |
| `build_fixture_fdr_map(fixtures, gws)` | `{(team_id, gw): avg_fdr_team}`. Double GWs average both fixtures. Teams with no fixture absent (blank GW). Always uses `fdr_team` (not `fdr_opp`). |
| `build_xp_matrix(predictions, fixtures, team_id_map, gws, sensitivity)` | Player × GW DataFrame of FDR-adjusted xP. Blank GW = 0. |
| `recommend_transfers(user_state, predictions, fixtures, horizon, …)` | Dispatcher: `horizon=1` → single-GW ILP; `horizon≥2` → multi-GW ILP. |
| `_recommend_single_gw(…)` | PuLP ILP. Variables: `x` (squad), `transfer_in/out`, `captain`, `hits`. Objective: `Σ xP·x + Σ xP·cap − 4·hits`. Budget constraint in 0.1M units. |
| `_recommend_multi_gw(…)` | Multi-period PuLP ILP over `horizon` GWs. FT carry-forward linearised with big-M=20. Bank tracked per GW. FDR weighting applied GW1+. |
| `recommend_wildcard(user_state, predictions)` | Calls `optimize_team(budget=user_state.total_value)` — unconstrained rebuild within current squad value. |
| `save_recommend_csv(plan, path, start_gw)` | Flattens per-GW transfer lists to CSV with columns: `gw, action, player_out, player_in, price_out, price_in, xp_out, xp_in, hit_cost, bank_after`. |

**Known limitations:**
- Multi-GW FDR weighting requires a `team_id_map` (team name → FPL team ID). Without bootstrap this mapping isn't built inside `_recommend_multi_gw`, so future GW xP falls back to raw xP (no FDR adjustment). The single-GW path also doesn't apply FDR today — it uses raw xP from predictions CSV.
- Captain in `_recommend_single_gw` is required to be in the squad (not strictly in the XI). This is a slight relaxation vs FPL rules but doesn't affect the result materially since the captain is always selected for the XI by the optimizer.

### Track A — Post-Match Analysis (`src/pipeline/analysis.py`)

| Function | What it does |
|----------|-------------|
| `compute_prediction_misses(picks_df, top_n=5)` | `actual_points − xP` per player, sorted by `abs(miss)` descending. Returns top N as list of dicts. |
| `compute_dream_team(live_data)` | Aliases `total_points → xP`, calls `select_xi()` on the full live player pool. No 3-per-club cap (matches FPL dream team rules). |
| `format_post_match_summary(…)` | Formats terminal output: your team vs recommended vs dream, benchmarks table, biggest misses. |
| `append_accuracy_log(path, gw, …)` | Appends one row to `results/accuracy_log.csv`. Columns: `gw, your_pts, your_predicted_xp, recommended_pts, recommended_xp, dream_team_pts, your_percentile_rank, best_score, top_1k/10k/100k/1m_score, avg_score, median_score, ranked_count, timestamp`. Creates file on first run. |

### Track A — Config & CLI (`src/config.py`, `user_config.example.yaml`, `src/pipeline/run.py`)

- `load_user_config(path)` / `UserConfigError` — validates `user_config.yaml`: requires `teams.default.entry_id` (int), optional `teams.alt`, `preferences.horizon_gws` (1–5), `max_hit_points`, `fdr_sensitivity`. Applies defaults for missing preference keys.
- New URL constants: `FPL_ENTRY_URL`, `FPL_EVENT_URL`, `FPL_LEAGUES_CLASSIC_URL`
- `optimize_team()` / `select_squad()` gain optional `budget: int | None` param (falls back to `SQUAD_RULES["budget"]` = 1000 units = £100M)
- `predict.save_full_predictions()` — writes `results/predictions_gw{N}.csv` with columns `element, code, name, position, team, xP, now_cost` (now_cost in 0.1M units)
- New CLI phases: `recommend` with flags `--horizon N`, `--wildcard`, `--team KEY`
- `phase_post_gw()` extended: after saving live data, loads predictions + user picks, computes dream team and benchmarks, prints summary, appends accuracy log — all skipped gracefully if `user_config.yaml` is missing

**Tests added:** 31 new tests (115 total, up from 84). All in `tests/test_recommend.py`, `tests/test_analysis.py`, and extensions to `tests/test_user.py`, `tests/test_run.py`.

---

## Improvement Ideas (Prioritised)

### P1 — High Impact, Low Effort

#### 1. Swap RF for XGBoost
- **What:** Replace `RandomForestRegressor` in `phase_retrain()` with `XGBRegressor`.
- **Why:** NB04 shows XGBoost MAE 1.026 vs RF 1.035 — consistent 1% improvement at
  no feature cost.
- **How:**
  ```python
  from xgboost import XGBRegressor
  model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
  ```
- **Files:** `src/pipeline/run.py:phase_retrain()`
- **Risk:** Low. Drop-in replacement; same joblib serialisation.

#### 2. Two-Stage Playing-Time Model
- **What:** Train a separate classifier to predict P(player plays ≥ 60 min), multiply
  by conditional points model output.
- **Why:** `minutes` is the dominant feature (16–30% importance). Predicting whether
  a player starts is the single biggest source of residual error.
- **How:** Stage 1 = `LogisticRegression` on `minutes_roll_4`, `selected_by_percent`,
  `availability_status`. Stage 2 = current RF/XGB on players where Stage 1 > 0.5.
- **Files:** new `src/pipeline/predict.py` (add `predict_playing_time()`)
- **Risk:** Medium. Requires labelled training data (binary minutes ≥ 60 target).

#### 3. Revive xG / xA Features (Understat)
- **What:** Fix `_original/data_collection/understat.py` — update hardcoded season
  `2024` to the current season and save xG/xA per player per GW into vaastav-like CSV.
- **Why:** NB04 shows these features would add signal; they were commented out only
  because the scraper was broken, not because the data was unhelpful.
- **How:**
  ```python
  # In understat.py, line ~45:
  # OLD: SEASON = "2024"
  # NEW: SEASON = str(datetime.now().year - (1 if datetime.now().month < 8 else 0))
  ```
  Then join `xG`, `xA` per player-GW in `prepare.build_merged_dataset()`.
- **Files:** `_original/data_collection/understat.py` (restore to `src/data_collection/`),
  `src/pipeline/prepare.py`
- **Risk:** Medium. Understat DOM may have changed; may need `playwright` instead of `requests`.

### P2 — Medium Impact, Medium Effort

#### 4. Positional Feature Engineering
- **What:** Add position-specific features that the global model ignores:
  - GK: `clean_sheets_roll_4`, `saves_roll_4`, `goals_conceded_roll_4`
  - DEF: `clean_sheets_roll_4`, `clearances`, `tackles`
  - FWD: `shots_on_target_roll_4`, `npxG_roll_4`
- **Why:** NB05 shows GK MAE is 0.770 vs FWD 1.249 — positions have very different
  point distributions that position-specific features could better capture.
- **Files:** `src/pipeline/features.py` (add `add_positional_features()`),
  `src/pipeline/predict.py` (split by position for inference)
- **Risk:** Medium. Requires defensive stats from FBref or live API (some available in
  `bootstrap` already: `goals_conceded`, `clean_sheets`).

#### 5. Sharpe Ratio / Risk-Adjusted Optimization
- **What:** Add an alternative optimisation objective that minimises variance subject
  to a minimum-points constraint, as in `_original/optimization/FPL_Sharpe.r`.
- **Why:** Pure xP maximisation consistently over-weights volatile attackers (Haaland
  or nothing). A risk-adjusted portfolio would be more robust to blank GWs.
- **How:**
  ```python
  # In optimize.py, add sharpe_optimize_team(predictions, historical_points_df):
  #   1. Build per-player variance from historical GW data.
  #   2. Objective: minimize sum(variance) subject to sum(xP) >= threshold.
  #   3. Use PuLP with a QP approximation or scipy.optimize.minimize.
  ```
- **Files:** `src/pipeline/optimize.py` (new function `risk_adjusted_optimize_team()`),
  `src/pipeline/run.py` (add `--strategy sharpe` CLI flag)
- **Risk:** Medium–High. True QP requires `cvxpy` or `scipy`; PuLP only handles LP/ILP.

#### 6. Player Clustering for Cold-Start (New-Season Players)
- **What:** For players with < 8 GW history (rolling features are NaN), assign them
  to a cluster and use the cluster centroid's xP as a prior.
- **Why:** NB03 shows 3-cluster KMeans gives MAE 1.327 — a 1.3% improvement over
  treating new players as zero. Currently the pipeline excludes these players entirely.
- **How:**
  ```python
  # In predict.py, after predict_next_gw():
  #   1. Flag rows where total_points_roll_8 is NaN (insufficient history).
  #   2. Cluster by [position, selected_by_percent, now_cost].
  #   3. Assign cluster centroid xP as prediction for cold-start players.
  ```
- **Files:** `src/pipeline/predict.py`, new `src/pipeline/clustering.py`
- **Risk:** Low. Fallback is xP from API (already implemented); clustering is additive.

### P3 — Low Impact or High Effort

#### 7. Pipeline Scheduling (Cron / GitHub Actions)
- **What:** Automate the `full` phase to run ~48h before each deadline; automate
  `post-gw` to run ~2h after final whistle on GW Saturday.
- **How:** GitHub Actions workflow on a schedule, or Windows Task Scheduler calling
  `scripts/weekly_run.sh`.
- **Files:** `.github/workflows/weekly-pipeline.yml` (new), `scripts/weekly_run.sh`
  (update paths)
- **Risk:** Low for local scheduler; medium for GitHub Actions (secrets management,
  API rate limits from Actions runners).

#### 8. FBref Defensive Stats
- **What:** Revive `_original/data_collection/fbref.py` to pull tackles, interceptions,
  blocks per player per GW for better DEF/MID feature engineering.
- **Why:** Defensive FPL returns are hard to predict without defensive-action data.
  Pipeline currently only uses `clean_sheets` as a defensive feature.
- **Risk:** High. FBref aggressively rate-limits scrapers and changes table structure.
  Consider using `understat-client` or a commercial data provider instead.

#### 9. Ensemble Predictions (RF + XGBoost)
- **What:** Average predictions from multiple models (RF global + XGB global +
  positional models) for final xP. Standard ensemble reduces variance.
- **Why:** NB04 shows models have similar error profiles but different failure modes.
- **Files:** `src/pipeline/predict.py` (add `ensemble_predict_next_gw()`)
- **Risk:** Low. All models already serialised; only inference code changes.

---

## Constraints the Optimizer Must Always Enforce

Extracted from `FPL.R` and `FPL_xPMin.R` — these were the hard constraints in the R solver:

```
Squad (15 players):
  - Total budget ≤ £100.0M (now_cost in tenths: ≤ 1000)
  - Exactly 2 GK
  - Exactly 5 DEF
  - Exactly 5 MID
  - Exactly 3 FWD
  - Max 3 players from any single club

XI (11 starters from squad):
  - Exactly 1 GK
  - At least 3 DEF
  - At least 2 MID
  - At least 1 FWD
  - Total starters = 11

Captain / Vice-Captain:
  - 1 player designated captain (xP × 2)
  - 1 player designated vice-captain (fallback if captain doesn't play)
```

All constraints are already implemented in `src/pipeline/optimize.py`. This section is
a reference for regression testing when the optimizer is modified.

---

## API Endpoints

```python
BASE_URL = "https://fantasy.premierleague.com/api/"

# Original (from _original/data_collection/getters.py) — all in src/pipeline/fetch.py
bootstrap  = BASE_URL + "bootstrap-static/"          # All player + team metadata
element    = BASE_URL + "element-summary/{id}/"      # Per-player GW history
fixtures   = BASE_URL + "fixtures/"                  # All fixtures with FDR
live       = BASE_URL + "event/{gw}/live/"           # Live GW points (mid-match)

# Added in Track A — all in src/pipeline/user.py via FPL_*_URL constants in src/config.py
entry      = BASE_URL + "entry/{id}/"                # Entry info: bank, league membership
picks      = BASE_URL + "entry/{id}/event/{gw}/picks/"  # GW squad picks + selling prices
history    = BASE_URL + "entry/{id}/history/"        # GW-by-GW history + transfer log
standings  = BASE_URL + "leagues-classic/{id}/standings/?page_standings={p}&event={gw}"
                                                     # Overall league standings (paginated, 50/page)
```

No authentication required. No published rate limits, but 3–5 s sleep between
player fetches is safe (700 players ≈ 35 min for full collection).
The standings endpoint is used to find score cutoffs at ranks 1k/10k/100k/1M.

---

## Notebook Observations Archive

Brief status of each archived notebook for future developers who want to revisit them.

| Notebook | What it does | Runnable? | Key blocker |
|----------|-------------|-----------|-------------|
| 01_eda.ipynb | Actually a data collector, not EDA. Hits FPL API for all 700+ players. 35+ min runtime. | No | Hardcoded Windows path; superseded by pipeline |
| 02_feature_engineering.ipynb | Builds rolling/momentum features. Has 3 competing implementations — the vectorised version (cell `80c05094`) is 100× faster than iterrows. | Partial | Assumes pre-built `cleaned_merged_seasons1.csv` (not in repo) |
| 03_player_clustering.ipynb | KMeans (K=3) on player stats for cold-start prior. MAE improvement: 1.556 → 1.327. | Partial | Depends on `team_key.csv` from NB07 |
| 04_model_training.ipynb | Global RF + XGBoost. XGBoost wins (MAE 1.026). Top feature: minutes (16–30%). | Yes (path fix) | Hardcoded season path |
| 05_model_training_positional.ipynb | Position-specific models. GK best (0.770), FWD worst (1.249). | Yes (path fix) | Hardcoded season path |
| 06_team_optimization.ipynb | MINLP/GEKKO exploration. **Incomplete.** R scripts are the real optimizer. | No | `gekko` unmaintained; no output |
| 07_team_key_mapping.ipynb | Builds team_key.csv for NB03. Small utility. | Yes (path fix) | Hardcoded path |

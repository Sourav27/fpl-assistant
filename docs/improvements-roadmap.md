# FPL Assistant — Improvements Roadmap

Distilled from archived research code (`_original/notebooks/`, `_original/optimization/`,
`_original/data_collection/`) before those folders were removed from version control.
Last updated: 2026-03-28.

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

## API Endpoints (from `_original/data_collection/getters.py`)

```python
BASE_URL = "https://fantasy.premierleague.com/api/"

bootstrap  = BASE_URL + "bootstrap-static/"       # All player + team metadata
element    = BASE_URL + "element-summary/{id}/"   # Per-player GW history
fixtures   = BASE_URL + "fixtures/"               # All fixtures with FDR
live       = BASE_URL + "event/{gw}/live/"        # Live GW points (mid-match)
```

No authentication required. No published rate limits, but 3–5 s sleep between
player fetches is safe (700 players ≈ 35 min for full collection).
All endpoints implemented in `src/pipeline/fetch.py`.

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

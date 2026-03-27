# Pipeline Run Observations — 2026-03-27

Full end-to-end pipeline test against vaastav/Fantasy-Premier-League dataset (10 seasons, 2016-2025).

---

## Environment

- Python 3.12.3, R 4.x
- Packages: pandas 3.0.1, scikit-learn 1.8.0, xgboost 3.2.0, lpSolve (R)
- Data: vaastav/Fantasy-Premier-League (shallow clone, all 10 seasons)

---

## Notebook 01 — EDA / Data Collection

**Status:** NOT RUNNABLE as-is

**Issues:**
1. **Misnomer:** This is a data collection notebook, not EDA. It hits the live FPL API endpoint (`/api/bootstrap-static/`, `/api/element-summary/{id}/`) for every player.
2. **Runtime:** ~35 minutes minimum (700+ players × 3-second sleep per request). The second cell block uses 1-second sleep for the remaining players.
3. **Fragile collection pattern:** If interrupted mid-run, data is lost. Two separate collection runs are stitched together manually (cells merge `all_history_df1` and `all_history_df2`).
4. **Hardcoded season:** Saves to `all_history_df_2223.csv` — season is baked into filenames.
5. **Not needed:** The vaastav dataset provides the same data already cleaned and merged.

**Recommendation:**
- Rename to `01_data_collection_api.ipynb` or deprecate entirely.
- Replace with a proper EDA notebook that explores the vaastav dataset directly.
- If API collection is needed for live predictions, refactor into `src/data_collection/` as a script with resume capability.

---

## Notebook 02 — Feature Engineering

**Status:** RUNS WITH FIXES (path fix + performance fix)

**Issues:**
1. **Hardcoded Windows path:** `D:\FPL\fpl-optimization\data\Fantasy-Premier-League\data\cleaned_merged_seasons1.csv` — must be relative.
2. **Missing input file:** Expects `cleaned_merged_seasons1.csv` which doesn't exist in the standard vaastav dataset. This was a custom merge the original team created locally but never committed. Had to recreate it by merging per-GW files with fixtures data to get `xP`, `fdr_team`, `fdr_opp_team`.
3. **Catastrophically slow:** The row-by-row iteration with `iterrows()` over 246K rows takes 30+ minutes. For each player-match, it filters the entire DataFrame to find previous matches.
4. **Three competing implementations:** The notebook contains three different versions of the same feature engineering logic (cells `027f3787`, `9c5e0580`, `0fa67bbb`), with the first two being slower and the third being incomplete.
5. **Missing columns in vaastav data:** `fdr_team`, `fdr_opp_team`, `transfers_net`, `fdr_net` don't exist in `cleaned_merged_seasons.csv`. They must be derived from per-GW files and fixtures.
6. **Encoding issues:** Seasons 2016-17 through 2018-19 have player names with accented characters (e.g., é) stored in latin-1, not UTF-8. Requires `encoding='latin-1'` parameter.

**Fix applied:**
- Vectorized approach using `pd.merge` with shifted match_count: completes in ~5 seconds instead of 30+ minutes.
- Created `cleaned_merged_seasons1.csv` from per-GW files with FDR data merged from fixtures.

**Recommendation:**
- Replace iterrows approach with vectorized merge (100x speedup).
- Create a data preparation script that builds `cleaned_merged_seasons1.csv` from the vaastav dataset, so the notebook doesn't depend on a file that was never committed.
- Remove duplicate implementations; keep only the vectorized version.

---

## Notebook 03 — Player Clustering

**Status:** RUNS WITH FIXES (path fix + column handling)

**Issues:**
1. **Hardcoded Windows path** to timeseries dataset.
2. **`dropna(axis=1)` drops too aggressively:** For cold-start players (match_count < 5), all lag columns are NaN, so dropping NaN columns removes nearly everything including `position`.
3. **Depends on `data/team_key.csv`** which is created by notebook 07 but needed here — ordering dependency issue.
4. **Season split hardcoded** to `2024-25` as test set (was originally `2022-23`).
5. **Low silhouette scores** (0.14-0.17): clusters aren't well-separated with the available features for cold-start players.
6. **NaN team entry:** Some players have missing team data, causing a NaN entry in team_key.csv.

**Results (with fixes):**
- Optimal K=3 clusters
- Clustering MAE: 1.327 (vs baseline 1.436) — modest improvement
- Cluster 0: 7011 players, avg 1.09 pts (bench/substitute players)
- Cluster 1: 7133 players, avg 1.00 pts (similar low-output group)
- Cluster 2: 901 players, avg 3.90 pts (starters with higher expected output)

**Recommendation:**
- Use more features: include `position_encoded` and `value` as primary clustering dimensions.
- Consider position-specific clustering rather than global clustering.
- The cold-start problem might be better solved with transfer-market heuristics than KMeans.

---

## Notebook 04 — Global Model Training

**Status:** RUNS SUCCESSFULLY

**Issues:**
1. **Hardcoded Windows path** to timeseries dataset.
2. **Hardcoded column indices** in some cells (e.g., referencing column by position number).
3. **XGBoost API usage:** Original notebook uses deprecated `xgb.DMatrix` / `xgb.train` / `xgb.cv` API. Modern XGBRegressor wrapper is cleaner.
4. **Feature importance:** `1_minutes` dominates (~16-30% importance) — model is largely predicting "will this player play?".

**Results:**
| Model | MAE | RMSE |
|---|---|---|
| Baseline (mean) | 1.556 | 2.372 |
| Linear Regression | 1.075 | 1.967 |
| Ridge (alpha=100) | 1.075 | 1.967 |
| Random Forest | 1.035 | 1.948 |
| XGBoost | **1.026** | **1.952** |

**Key observations:**
- XGBoost slightly outperforms RF (MAE 1.026 vs 1.035).
- OOB R² for RF is 0.313 — model explains ~31% of variance, which is typical for per-GW point prediction.
- Top features across both models: `1_minutes`, `1_ict_index`, `1_total_points`, `1_bps`, `transfers_in`.
- The model is trained on all seasons except 2022-23 and tested on 2022-23.

**Recommendation:**
- Consider time-series cross-validation instead of single-season holdout.
- Feature engineering: add rolling averages (the original notebook had commented-out Understat/FBref features like xG, xA that would likely improve predictions).
- The heavy reliance on `1_minutes` suggests a two-stage model might work better: first predict if a player will play, then predict points conditional on playing.

---

## Notebook 05 — Positional Model Training

**Status:** RUNS SUCCESSFULLY

**Results by position:**
| Position | Train Size | RF MAE | RF OOB R² | XGB MAE |
|---|---|---|---|---|
| GK | 13,736 | 0.770 | 0.438 | 0.789 |
| DEF | 41,003 | 1.102 | 0.273 | 1.105 |
| MID | 52,469 | 0.980 | 0.327 | 0.983 |
| FWD | 14,890 | 1.249 | 0.321 | 1.270 |

**Key observations:**
- GK models have highest R² (0.438) — goalkeeper performance is more predictable (clean sheet-driven).
- FWD models have highest MAE (1.249) — forward points are most volatile (goal-dependent).
- Positional models don't consistently beat the global model — suggests the global model already captures positional differences through the `position_encoded` feature.
- RF slightly outperforms XGBoost for all positions.

**Recommendation:**
- Positional models show marginal benefit. Consider an ensemble approach instead.
- FWD model could benefit from xG/xA features (currently missing from the feature set).

---

## Notebook 06 — Team Optimization

**Status:** INCOMPLETE / EXPLORATORY

**Issues:**
1. Reads a single GW file (`datasets/gw1.csv`) — not a full pipeline step.
2. Imports `gekko` (MINLP solver) which is not in requirements.txt.
3. Contains only exploration code (random selection vectors, GEKKO demo).
4. Does not produce any output used downstream.

**Recommendation:**
- Either flesh out into a proper Python optimization notebook (replacing R scripts) or remove from the pipeline and mark the R scripts as the optimization step.
- If keeping, switch from GEKKO to PuLP or scipy.optimize.linprog for consistency with the LP approach.

---

## Notebook 07 — Team Key Mapping

**Status:** RUNS WITH PATH FIX

**Issues:**
1. **Hardcoded Windows path** to timeseries dataset.
2. **Ordering dependency:** Creates `team_key.csv` needed by notebook 03, but is numbered after it.
3. **NaN team:** Some players have missing team data, producing a NaN entry.

**Recommendation:**
- Move team key generation into the data preparation step or into notebook 02.
- Filter out NaN teams before saving.

---

## R Optimization Scripts

**Status:** FPL_xPMin.R RUNS WITH FIXES

**Issues (all R scripts):**
1. **Hardcoded Windows paths:** All scripts use `setwd("C:\\Users\\debna\\OneDrive - ...")` and `datasets\\` paths.
2. **Hardcoded column indices:** Position dummy variables are referenced by column number (e.g., `df1[,49]` for GK) — breaks if column order changes.
3. **Hardcoded team names:** Constraint matrix rows are hardcoded to 2022-23 team names. Different seasons have different teams (promotion/relegation).
4. **Missing GW 7:** Season 2022-23 has no GW 7 (blank gameweek), causing index issues if not handled.
5. **Edge cases:** GW 12 and GW 36 produced 0 xP — likely due to data quality issues (all players having 0 xPMin).
6. **`for(k in 1:1)`:** The xPMin script was hardcoded to only run GW 1 — the loop range was manually limited.

**Results (FPL_xPMin, dynamically fixed):**
- Successfully optimized 33 gameweeks (GW 5-38, excluding GW 7)
- Average predicted xP per GW: 98.9 points (with captain)
- Average actual points per GW: 95.0 points
- Best GW: GW 29 — 148.1 xP predicted, 146 actual (Ollie Watkins captain)
- Erling Haaland selected as captain in 15/33 GWs

**Recommendation:**
- Migrate to Python (PuLP). This eliminates the dual-language dependency and hardcoded column indices.
- Make team list dynamic (read from data rather than hardcoded).
- Add proper handling for blank/double gameweeks.

---

## Cross-Cutting Issues

### 1. Hardcoded Paths (ALL notebooks + R scripts)
Every notebook and R script has hardcoded absolute Windows paths. Must be converted to relative paths.

**Files affected:**
- `notebooks/02_feature_engineering.ipynb` — `D:\FPL\...`
- `notebooks/03_player_clustering.ipynb` — `D:\FPL\...`
- `notebooks/04_model_training.ipynb` — `D:\FPL\...` (likely)
- `notebooks/05_model_training_positional.ipynb` — `D:\FPL\...` (likely)
- `notebooks/06_team_optimization.ipynb` — `datasets/...`
- `notebooks/07_team_key_mapping.ipynb` — `D:\FPL\...`
- All R scripts in `src/optimization/` — `C:\Users\debna\...`

### 2. Missing Data Preparation Step
The pipeline assumes `cleaned_merged_seasons1.csv` exists, but this file must be constructed from the vaastav dataset. Need a `src/data_preparation.py` script that:
- Merges per-GW files across seasons (with latin-1 encoding for older seasons)
- Joins fixture difficulty ratings from fixtures.csv
- Adds derived columns (fdr_net, transfers_net)
- Outputs `cleaned_merged_seasons1.csv`

### 3. Feature Engineering Performance
The iterrows-based approach is O(n²) per player. Vectorized merge approach is O(n log n). This is the single biggest technical improvement available — 100x speedup.

### 4. Missing Advanced Features
The notebook has commented-out code for Understat xG/xA features and FBref defensive stats. These would likely improve model accuracy significantly but require:
- Web scraping scripts to be functional
- Date-matching logic between Understat/FBref and FPL data
- These data sources are fragile (scraping can break with site changes)

### 5. No Pipeline Orchestration
There's no way to run the pipeline end-to-end. Need either:
- A `Makefile` or `run_pipeline.py` script
- Or convert notebooks to `.py` scripts with proper imports

### 6. Model Persistence
Models are saved as `.sav` files (joblib). No versioning, no metadata about training data or hyperparameters.

---

## Prioritized Improvement List

### High Priority (Functionality)
1. **Fix all hardcoded paths** — convert to relative paths using `os.path` or `pathlib`
2. **Create data preparation script** — `src/data_preparation.py` to build `cleaned_merged_seasons1.csv`
3. **Fix notebook execution order** — move team key mapping before clustering (or into data prep)
4. **Add requirements for missing packages** — `gekko` if keeping NB06, or remove the import

### High Priority (Performance)
5. **Vectorize feature engineering** — replace iterrows with merge-based approach (100x speedup)
6. **Remove duplicate code** — NB02 has 3 versions of the same logic

### Medium Priority (Quality)
7. **Migrate R to Python** — replace lpSolve with PuLP for single-language pipeline
8. **Add pipeline runner** — `Makefile` or `run_pipeline.py` for end-to-end execution
9. **Add time-series cross-validation** — instead of single-season holdout
10. **Add Understat/FBref features** — xG, xA, defensive stats (the commented-out code)

### Low Priority (Nice to have)
11. **Two-stage prediction model** — predict playing time first, then points|playing
12. **Position-specific feature sets** — different features for GK vs FWD
13. **Ensemble models** — combine global + positional predictions
14. **Model versioning** — save metadata with trained models
15. **Create proper EDA notebook** — replace the current NB01 (which is data collection)

---

## Summary

The pipeline is functional end-to-end with path fixes and data preparation. The core ML approach (RF/XGBoost on lagged player stats) works and produces reasonable predictions (MAE ~1.0 per player per GW). The R optimization correctly selects teams that score ~95 actual points per GW on average.

The biggest wins are:
1. Fix paths (5 min of work, unblocks everything)
2. Vectorize feature engineering (already proven: 30 min → 5 seconds)
3. Migrate R to Python (eliminates dual-language complexity)
4. Add missing xG/xA features (likely biggest accuracy improvement)

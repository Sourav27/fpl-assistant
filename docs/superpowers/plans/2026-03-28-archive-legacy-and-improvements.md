# Archive Legacy Code & Document Improvements Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Archive `notebooks/`, `src/optimization/`, and `src/data_collection/` into `_original/` (already git-ignored), then produce a standalone improvements roadmap that captures the research learnings from those files.

**Architecture:** Move legacy folders into the existing git-ignored `_original/` directory so they disappear from version control but remain on disk for reference. Then write `docs/improvements-roadmap.md` that distills every actionable insight. Finally update `CLAUDE.md` so it reflects the new directory layout.

**Tech Stack:** bash `mv`, Python (no new code), Markdown

---

## File Structure

| Action | Path | Notes |
|--------|------|-------|
| Move | `notebooks/` → `_original/notebooks/` | 7 Jupyter notebooks; already hardcoded-path-broken |
| Move | `src/optimization/` → `_original/optimization/` | 10 R scripts; superseded by PuLP |
| Move | `src/data_collection/` → `_original/data_collection/` | 10 Python scripts; FPL API layer superseded by `src/pipeline/fetch.py` |
| Create | `docs/improvements-roadmap.md` | All learnings + prioritised improvement backlog |
| Modify | `CLAUDE.md` | Remove references to deleted dirs; add pointer to roadmap |

---

## Task 1: Move `notebooks/` to `_original/`

**Files:**
- Move: `notebooks/` → `_original/notebooks/`

- [ ] **Step 1: Verify `_original/` exists and check current contents**

```bash
ls _original/
```

Expected: `data/` directory with archived CSVs. No `notebooks/` subfolder yet.

- [ ] **Step 2: Move the notebooks directory**

```bash
mv notebooks/ _original/notebooks/
```

- [ ] **Step 3: Verify the move**

```bash
ls _original/notebooks/
# Expected: 01_eda.ipynb  02_feature_engineering.ipynb  03_player_clustering.ipynb
#           04_model_training.ipynb  05_model_training_positional.ipynb
#           06_team_optimization.ipynb  07_team_key_mapping.ipynb
ls notebooks/ 2>&1  # Should say: ls: cannot access 'notebooks/': No such file or directory
```

- [ ] **Step 4: Confirm git no longer tracks the folder**

```bash
git status
# _original/ is in .gitignore so notebooks should not appear as deleted files
```

Expected: `notebooks/` files appear as deleted from git tracking (if they were tracked), or nothing if already ignored. Since notebooks were tracked, they will show as deleted — that's expected; we commit the deletion in Task 4.

- [ ] **Step 5: Commit the notebook removal**

```bash
git add -A notebooks/
git commit -m "$(cat <<'EOF'
chore: archive notebooks/ to _original/ (research-only, not production)

Notebooks are path-hardcoded and not runnable without manual fixes.
All production logic has been extracted into src/pipeline/.
Preserved in _original/ for reference; git-ignored.
EOF
)"
```

---

## Task 2: Move `src/optimization/` to `_original/`

**Files:**
- Move: `src/optimization/` → `_original/optimization/`

- [ ] **Step 1: Move the R scripts**

```bash
mv src/optimization/ _original/optimization/
```

- [ ] **Step 2: Verify**

```bash
ls _original/optimization/
# Expected: Covariance_TotalPoints.R  FPL.R  FPL_SC.r  FPL_Sharpe.r  FPL_best.r
#           FPL_xP.r  FPL_xPMin.R  FPLv2.r  lpSolveSample.R  xMin.R
```

- [ ] **Step 3: Commit**

```bash
git add -A src/optimization/
git commit -m "$(cat <<'EOF'
chore: archive src/optimization/ R scripts to _original/

R/lpSolve optimizer superseded by src/pipeline/optimize.py (PuLP/Python).
FPL_xPMin results documented in docs/improvements-roadmap.md.
Preserved in _original/optimization/ for reference.
EOF
)"
```

---

## Task 3: Move `src/data_collection/` to `_original/`

**Files:**
- Move: `src/data_collection/` → `_original/data_collection/`

Note: Before moving, verify none of `src/pipeline/` imports from `src/data_collection/`. A quick grep confirms the pipeline does not import from this module (it reimplements the FPL API calls in `fetch.py`).

- [ ] **Step 1: Check for live imports from the pipeline**

```bash
grep -r "data_collection" src/pipeline/ tests/
# Expected: no matches — pipeline is self-contained
```

- [ ] **Step 2: Move the data collection scripts**

```bash
mv src/data_collection/ _original/data_collection/
```

- [ ] **Step 3: Verify**

```bash
ls _original/data_collection/
# Expected: __init__.py  cleaners.py  collector.py  fbref.py  gameweek.py
#           getters.py  mergers.py  parsers.py  understat.py  utility.py
```

- [ ] **Step 4: Run tests to confirm nothing broke**

```bash
python -m pytest tests/ -q --tb=short
# All tests should still pass
```

- [ ] **Step 5: Commit**

```bash
git add -A src/data_collection/
git commit -m "$(cat <<'EOF'
chore: archive src/data_collection/ to _original/

FPL API layer superseded by src/pipeline/fetch.py.
Understat/FBref scrapers broken (hardcoded season 2024, stale paths).
Patterns documented in docs/improvements-roadmap.md for future revival.
EOF
)"
```

---

## Task 4: Write `docs/improvements-roadmap.md`

**Files:**
- Create: `docs/improvements-roadmap.md`

This document captures every actionable research finding extracted from the archived code before it leaves version control.

- [ ] **Step 1: Create the roadmap document**

Create `docs/improvements-roadmap.md` with the content below.

```markdown
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
```

- [ ] **Step 2: Verify the file was written correctly**

```bash
wc -l docs/improvements-roadmap.md
# Should be ~200+ lines
head -5 docs/improvements-roadmap.md
```

- [ ] **Step 3: Commit the roadmap**

```bash
git add docs/improvements-roadmap.md
git commit -m "$(cat <<'EOF'
docs: add improvements roadmap distilled from archived research code

Captures ML baselines, top features, R optimizer results (95 pts/GW avg),
9 prioritised improvement ideas, FPL constraints, and API endpoint reference.
EOF
)"
```

---

## Task 5: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

The existing CLAUDE.md still references `notebooks/`, `src/optimization/`, and
`src/data_collection/` as if they're present. Update the directory structure block
and any references.

- [ ] **Step 1: Open CLAUDE.md and identify the sections to update**

Look for:
- The directory tree under `## Directory Structure`
- Any mention of `notebooks/`, `src/data_collection/`, `src/optimization/`
- The `## Pipeline Workflow` section referencing notebooks

- [ ] **Step 2: Apply the following changes to CLAUDE.md**

Replace the directory tree to reflect the new layout:

```diff
-├── notebooks/          # 7 Jupyter notebooks numbered in pipeline order
 ├── src/
-│   ├── data_collection/  # 9 Python scripts — scrape FPL API, Understat, FBref
-│   └── optimization/     # 10 R scripts — lpSolve-based team selection
+│   └── pipeline/         # Weekly production pipeline (Python/PuLP)
+├── _original/          # Archived research code (git-ignored):
+│   ├── notebooks/      #   7 Jupyter notebooks (research history)
+│   ├── optimization/   #   10 R scripts (lpSolve; superseded by PuLP)
+│   └── data_collection/#   10 Python scripts (FPL/Understat/FBref scrapers)
```

In the `## Pipeline Workflow` section, replace the notebook table with a pointer:

```diff
-The notebooks are numbered to reflect execution order:
-
-| Step | Notebook | Description |
-...
+The original research notebooks and R optimizer are archived in `_original/`
+(git-ignored). See `docs/improvements-roadmap.md` for learnings and improvement ideas.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: update CLAUDE.md — remove archived folder references

notebooks/, src/optimization/, src/data_collection/ moved to _original/.
Added pointer to docs/improvements-roadmap.md for research learnings.
EOF
)"
```

---

## Validation Checklist

After all tasks complete, verify the following:

- [ ] `ls notebooks/` → "No such file or directory"
- [ ] `ls src/optimization/` → "No such file or directory"
- [ ] `ls src/data_collection/` → "No such file or directory"
- [ ] `ls _original/` → shows `notebooks/  optimization/  data_collection/  data/`
- [ ] `python -m pytest tests/ -q` → all tests pass (pipeline unaffected)
- [ ] `cat docs/improvements-roadmap.md | wc -l` → ≥ 150 lines
- [ ] `git log --oneline -5` → shows 4 new archival commits + 1 roadmap commit
- [ ] `cat CLAUDE.md | grep notebooks` → only reference is the `_original/` entry

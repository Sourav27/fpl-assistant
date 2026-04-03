# Track B — Fixture-Aware Per-Position Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single global RF model with 4 per-position RF models (GK/DEF/MID/FWD) trained with fixture-aware features (xGC, opponent form, DGW rest days) to improve rank correlation (Spearman ρ).

**Architecture:** We move from a flat prediction to a per-fixture prediction for each player. Data is enriched with opponent-side defensive stats. `predict.py` routes players to position-specific models and aggregates DGW results.

**Tech Stack:** Python, pandas, scikit-learn (RandomForest), PuLP.

---

### Task 1: Config & Constants Setup (B-F5)

**Files:**
- Modify: `src/config.py`
- Modify: `src/pipeline/predict.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Update `src/config.py`**
  Replace `ACTIVE_MODEL` with `ACTIVE_MODELS` dict.
  
```python
# src/config.py
ACTIVE_MODELS = {
    "GK": MODELS_DIR / "rf_gk_gw31.sav",
    "DEF": MODELS_DIR / "rf_def_gw31.sav",
    "MID": MODELS_DIR / "rf_mid_gw31.sav",
    "FWD": MODELS_DIR / "rf_fwd_gw31.sav",
}
# Keep ACTIVE_MODEL for backward compatibility during transition if needed
ACTIVE_MODEL = MODELS_DIR / "rf_model_gw31.sav"
```

- [ ] **Step 2: Update `predict.py:load_model`**
  Modify to accept position and handle missing models gracefully (return None or raise specific error for fallback).

- [ ] **Step 3: Run `pytest tests/test_config.py`**

- [ ] **Step 4: Commit**
```bash
git add src/config.py src/pipeline/predict.py
git commit -m "feat: setup ACTIVE_MODELS config for per-position models"
```

---

### Task 2: Data Preparation — Team Stats (B-F2)

**Files:**
- Modify: `src/pipeline/prepare.py`
- Test: `tests/test_prepare.py`

- [ ] **Step 1: Write `_add_team_rolling_stats` in `prepare.py`**
  Compute team-level defensive quality from player data.
  
```python
def _add_team_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    # Group by team, season, GW
    # sum(goals_conceded) -> team_GC
    # mean(total_points) allowed to opponent -> team_pts_allowed
    # rolling(4) of these
    ...
```

- [ ] **Step 2: Integrate into `build_merged_dataset`**
  Join these stats back to the player rows based on `opponent_team`.

- [ ] **Step 3: Run `pytest tests/test_prepare.py`**

- [ ] **Step 4: Commit**
```bash
git add src/pipeline/prepare.py
git commit -m "feat: add opponent defensive stats (xGC) to merged dataset"
```

---

### Task 5: Feature Engineering — Fixture Aware (B-F1)

**Files:**
- Modify: `src/pipeline/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Update `engineer_features` to include fixture features**
  - `is_home`: from `was_home`
  - `fixture_count`: count of fixtures per player per GW
  - `rest_days`: diff between kickoff times in DGW
  - `is_fixture_2`: boolean for second game in DGW

- [ ] **Step 2: Run `pytest tests/test_features.py`**

- [ ] **Step 3: Commit**
```bash
git add src/pipeline/features.py
git commit -m "feat: engineer fixture-aware features (DGW, rest days, home/away)"
```

---

### Task 4: Positional Routing & DGW Aggregation (B-F3, B-F4)

**Files:**
- Modify: `src/pipeline/predict.py`
- Test: `tests/test_predict.py`

- [ ] **Step 1: Update `predict_next_gw`**
  - Split players into per-fixture rows.
  - Route each row to the correct position model.
  - If model missing, use fallback logic.
  - Sum results for DGW players.

- [ ] **Step 2: Run `pytest tests/test_predict.py`**

- [ ] **Step 3: Commit**
```bash
git add src/pipeline/predict.py
git commit -m "feat: implement positional routing and DGW prediction aggregation"
```

---

### Task 5: Retrain Overhaul — 4 Models (B-F6)

**Files:**
- Modify: `src/pipeline/run.py`
- Test: `tests/test_run.py`

- [ ] **Step 1: Modify `phase_retrain`**
  - Iterate through positions: GK, DEF, MID, FWD.
  - Filter training set by position.
  - Train and save 4 separate `.sav` files.

- [ ] **Step 2: Run `pytest tests/test_run.py`**

- [ ] **Step 3: Commit**
```bash
git add src/pipeline/run.py
git commit -m "feat: retrain phase now builds 4 per-position models"
```

---

### Task 6: Spearman ρ Metric (B-F7)

**Files:**
- Modify: `src/pipeline/analysis.py`
- Modify: `src/pipeline/run.py`
- Test: `tests/test_analysis.py`

- [ ] **Step 1: Implement `compute_spearman_rho` in `analysis.py`**
  Using `scipy.stats.spearmanr`.

- [ ] **Step 2: Update `append_accuracy_log`**
  Add `spearman_rho` column to `accuracy_log.csv`.

- [ ] **Step 3: Run `pytest tests/test_analysis.py`**

- [ ] **Step 4: Commit**
```bash
git add src/pipeline/analysis.py src/pipeline/run.py
git commit -m "feat: add Spearman rank correlation (rho) to accuracy logging"
```

---

### Task 7: Verification

- [ ] **Step 1: Run `python -m src.pipeline.run retrain --gw 31`**
- [ ] **Step 2: Run `python -m src.pipeline.run predict --gw 32`**
- [ ] **Step 3: Verify results/predictions_gw32.csv contains expected xP variance**
- [ ] **Step 4: Update CLAUDE.md status**

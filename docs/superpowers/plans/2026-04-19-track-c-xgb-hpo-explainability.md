# Track C — XGBoost HPO & SHAP Explainability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed RF hyperparameters with Optuna-tuned RF and XGBoost models (per-position winner selected by walk-forward Spearman ρ), and add SHAP-based explainability so every player recommendation is backed by the top feature drivers.

**Architecture:** Two independent sub-tracks — (1) HPO: `phase_retrain` grows an Optuna tuning loop that trains both RF and XGB per position using time-series-safe CV, benchmarks on the current season via `promote.py`, and promotes the per-position winner; (2) Explainability: `predict.py` gains a `compute_shap_reasons()` function that attaches the top-3 SHAP feature contributions to every player row, surfaced in `predictions.csv` and `recommend.csv`.

**Tech Stack:** `optuna` (Bayesian TPE), `shap>=0.44.0,<1.0` (TreeExplainer — zero extra inference cost for tree models), `xgboost` (already in requirements), `scikit-learn` RF + `TimeSeriesSplit` (existing), `scipy.stats.spearmanr` (existing).

**New dependencies:** `optuna>=3.0.0`, `shap>=0.44.0,<1.0` — add to `requirements.txt`.

**MLOps cycle covered:**
- Data → **Feature enrichment** (Task 0 — 3 high-ROI features grounded in FPL scoring rules)
- **Data quality gate** (Task 1 — validates before any training)
- Train → Tune (Optuna HPO with `TimeSeriesSplit`, Task 2–3)
- Evaluate → Promote (walk-forward ρ gate + live ρ guard, existing `promote.py` + Task 4)
- Serve → Explain (SHAP at prediction time, Task 5–6)
- Monitor (accuracy_log Spearman ρ already tracked; live ρ trend used in promotion gate)

**Critical deployment note:** Tasks 3 and 4 are coupled — Task 3 introduces `algorithm=None` in `run.py` which breaks `promote.py` until Task 4's fix lands. Both must be completed before merging to master. Never commit Task 3 alone.

---

## Feature Alignment Summary (from data-scientist review)

The data-scientist agent audited current features against FPL scoring rules. Three highest-ROI additions with no new data dependencies:

| Priority | Feature | Positions | Why | Source |
|---|---|---|---|---|
| 1 | `opponent_xg_for_roll_4` | GK, DEF | Fixes DEF ρ=0.613 — CS is 4 pts but model has no forward-looking opponent attack signal | Aggregate `expected_goals` per opponent team per GW from vaastav — no new data. **Not** `xGC_rolling_4` (that is goals conceded = defensive quality, not attacking threat) |
| 2 | `penalty_taker` flag | MID, FWD | Highest per-event variance; separates Salah/Bruno from peers — already in your bootstrap JSON | FPL API bootstrap `penalties_order` field — already collected in every snapshot |
| 3 | `saves_roll_4` | GK | Entirely unmodelled scoring stream (~1.3 pts/GW for busy GKs; 3 saves = 1 pt) | vaastav `gws/saves` column |
| 4 | `is_cb` flag | DEF | Model conflates CB/FB — CBs score via CS, FBs via assists | **Deferred** — position is a manager decision that changes match-to-match; needs external positional data (e.g. FBref) |
| Drop | `opponent_form_rolling_6`, `is_fixture_2` | All | Zero importance confirmed in Track B | — |

Task 0 implements priorities 1–3. Priority 4 (`is_cb`) deferred pending data source research.

### Defensive contributions — research note (TODO)

New 2025-26 FPL rule: DEF ≥10 CBI+tackles = 2 pts; MID/FWD ≥12 CBI+tackles+recoveries = 2 pts. This is an entirely new scoring stream with no historical analogue.

**Data gap:** vaastav `gws/` does NOT include `clearances_blocks_interceptions`, `tackles`, or `recoveries` columns (confirmed by column audit). The FPL API element-summary also does not expose raw CBI counts — it records the bonus points that result but not the underlying stats.

**Action required before implementing this feature:**
1. Check whether the 2025-26 FPL API `element-summary/{id}/history` endpoint now exposes `clearances`, `tackles`, `recoveries` for the current season.
2. If yes: collect from live API for 2025-26; use `bps_roll_4` as a proxy for all historical seasons (BPS sub-events partially encode CBI contributions).
3. If no: source from FBref or StatsBomb match data — but note Track H explicitly dropped FBref due to rate limiting and DOM instability.

**Interim approach (this plan):** `bps_roll_4` is the best available proxy — BPS rewards CBI via its sub-event scoring, so players who regularly earn CBIT points will have elevated rolling BPS. This is already in the model. Add a `TODO: replace bps_roll_4 with explicit cbi_roll_4 once data source confirmed` comment in `features.py`.

This research is tracked as a future Track D/E task, not Track C.

---

## File Map

| File | Change |
|------|--------|
| `requirements.txt` | Add `optuna>=3.0.0`, `shap>=0.44.0,<1.0` |
| `src/pipeline/tune.py` | **NEW** — `validate_training_data()` + `tune_position_model()` with TimeSeriesSplit CV |
| `src/pipeline/run.py` | Replace manual RF fit in `phase_retrain` with `validate_training_data()` + `tune_position_model()`; pass `algorithm=None` to `run_promotion_pipeline`; add `--n-trials`/`--algos` flags; add live-ρ guard before promotion |
| `src/pipeline/promote.py` | Fix `run_promotion_pipeline` to infer algo from path stem into `result_entry["algorithm"]`; validate inferred algo; fix `build_active_models_manifest` to read per-position algo from ledger |
| `src/pipeline/predict.py` | Add `compute_shap_reasons()` with feature-column validation; call from `predict_next_gw_per_position`; add `"shap_reason": "first"` to DGW aggregation |
| `results/2025-26/gw{N}/predictions.csv` | New column: `shap_reason` (pipe-separated top-3 features with signed contributions) |
| `results/2025-26/gw{N}/recommend.csv` | `shap_reason` column for each transferred-in player |
| `tests/test_tune.py` | **NEW** — unit tests for `validate_training_data` + `tune_position_model` |
| `tests/test_shap_explain.py` | **NEW** — unit tests for `compute_shap_reasons` |
| `tests/test_predict_position.py` | Extend: verify `shap_reason` column present, DGW aggregation preserves it |
| `tests/test_promote.py` | Extend: verify per-position algo written to ledger + manifest |
| `tests/test_run.py` | Extend: verify `tune_position_model` called; live-ρ guard blocks promotion |

---

## Task 0: Feature enrichment — 3 high-ROI additions grounded in FPL scoring rules

**Files:**
- Modify: `src/pipeline/features.py` — add `saves_roll_4`, `opponent_xg_for_roll_4`, drop `opponent_form_rolling_6` + `is_fixture_2`
- Modify: `src/pipeline/prepare.py` — join opponent's xGC at fixture time for `opponent_xg_for_roll_4`; extract `penalty_taker` from bootstrap
- Modify: `src/pipeline/predict.py` — add new cols to `ALL_FEATURE_COLUMNS`; remove dropped cols
- Modify: `tests/test_features.py` — tests for new features
- Modify: `tests/test_prepare_opponent_stats.py` — tests for opponent xG join

- [ ] **Step 0.1: Write failing tests for `saves_roll_4`**

```python
# tests/test_features.py — add:
def test_saves_roll_4_computed(sample_gw_df):
    """saves_roll_4 must be a 4-GW rolling mean of the saves column."""
    # sample_gw_df must have 'saves' column with values
    sample_gw_df["saves"] = [3, 4, 2, 5, 1, 3, 0, 2]  # adapt to fixture shape
    result = engineer_features(sample_gw_df)
    assert "saves_roll_4" in result.columns
    assert result["saves_roll_4"].notna().any()
```

- [ ] **Step 0.2: Write failing test for `opponent_xg_for_roll_4`**

```python
# tests/test_prepare_opponent_stats.py — add:
def test_opponent_xg_for_roll_4_joined(merged_df):
    """Each player row must have opponent_xg_for_roll_4 = opponent team's xGC_rolling_4."""
    result = build_merged_dataset(...)  # or call the join helper directly
    assert "opponent_xg_for_roll_4" in result.columns
    # Two players facing same opponent in same GW must have same opponent_xg_for_roll_4
    gw_group = result[result["GW"] == 5]
    opp_vals = gw_group.groupby("opponent_team")["opponent_xg_for_roll_4"].nunique()
    assert (opp_vals == 1).all(), "All players vs same opponent should share opponent_xg_for_roll_4"
```

- [ ] **Step 0.3: Write failing test for `penalty_taker`**

```python
# tests/test_features.py — add:
def test_penalty_taker_binary(sample_players_df):
    """penalty_taker must be 1 for penalties_order==1, 0 otherwise."""
    sample_players_df["penalties_order"] = [1, 2, None, 1, 3]
    result = engineer_features(sample_players_df)
    assert "penalty_taker" in result.columns
    assert result.loc[result["penalties_order"] == 1, "penalty_taker"].eq(1).all()
    assert result.loc[result["penalties_order"] != 1, "penalty_taker"].eq(0).all()
```

- [ ] **Step 0.4: Run to confirm failures**

```bash
python -m pytest tests/test_features.py -k "saves_roll_4 or penalty_taker" \
               tests/test_prepare_opponent_stats.py -k "opponent_xg_for" -v
```

- [ ] **Step 0.5: Add `saves_roll_4` to `features.py`**

In `engineer_features`, in the rolling-feature construction block, add `saves` alongside the existing rolling columns:

```python
# Wherever total_points_roll_4 is computed, add:
for col in ["saves"]:
    if col in df.columns:
        df[f"{col}_roll_4"] = (
            df.groupby(["code", "season"])[col]
            .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        )
```

- [ ] **Step 0.6: Add `penalty_taker` to `features.py`**

```python
# After availability scaling block in engineer_features:
if "penalties_order" in df.columns:
    df["penalty_taker"] = (df["penalties_order"] == 1).astype(int)
else:
    df["penalty_taker"] = 0
```

- [ ] **Step 0.7: Add `opponent_xg_for_roll_4` in `prepare.py`**

**Source:** vaastav `gws/` has `expected_goals` per player per GW. Aggregate per team to get team-level xG-for, then join from the opponent's perspective. Do NOT use `xGC_rolling_4` (that is goals conceded = defensive quality, not attacking threat).

Read `prepare.py` first to identify the exact column name for opponent team (`opponent_team` or similar) before writing this code.

```python
# After building the merged player-GW dataframe in prepare.py / features.py:

# Step 1: compute team-level xG-for rolling avg from player expected_goals
team_xg = (
    merged
    .groupby(["team", "season", "GW"], as_index=False)["expected_goals"]
    .sum()
    .rename(columns={"expected_goals": "team_xg_for"})
)
team_xg["xg_for_roll_4"] = (
    team_xg.groupby(["team", "season"])["team_xg_for"]
    .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
)

# Step 2: join as opponent feature
opp_xg = team_xg[["team", "season", "GW", "xg_for_roll_4"]].rename(columns={
    "team": "opponent_team",   # adapt to actual column name in prepare.py
    "xg_for_roll_4": "opponent_xg_for_roll_4",
})
merged = merged.merge(
    opp_xg, on=["opponent_team", "season", "GW"], how="left"
)
```

- [ ] **Step 0.8: Remove dropped features from `predict.py`**

In `ALL_FEATURE_COLUMNS` (and `FIXTURE_FEATURE_COLUMNS` in `features.py`), remove `opponent_form_rolling_6` and `is_fixture_2`. Add `saves_roll_4`, `opponent_xg_for_roll_4`, `penalty_taker`.

- [ ] **Step 0.9: Run full test suite**

```bash
python -m pytest tests/ -q
```
Expected: all pass (new feature tests pass; no regressions in existing tests).

- [ ] **Step 0.10: Commit**

```bash
git add src/pipeline/features.py src/pipeline/prepare.py src/pipeline/predict.py \
        tests/test_features.py tests/test_prepare_opponent_stats.py
git commit -m "feat(track-c): add saves_roll_4, opponent_xg_for_roll_4, penalty_taker; drop zero-importance features"
```

---

## Task 1: Dependencies & `tune.py` with data validation

**Files:**
- Modify: `requirements.txt`
- Create: `src/pipeline/tune.py`
- Create: `tests/test_tune.py`

- [ ] **Step 1.1: Add dependencies to `requirements.txt`**

```
# After xgboost line:
optuna>=3.0.0
shap>=0.44.0,<1.0
```

- [ ] **Step 1.2: Install**

```bash
pip install "optuna>=3.0.0" "shap>=0.44.0,<1.0"
```

- [ ] **Step 1.3: Write failing tests**

```python
# tests/test_tune.py
import numpy as np
import pandas as pd
import pytest
from src.pipeline.tune import tune_position_model, validate_training_data

RNG = np.random.default_rng(42)
N = 300

def _make_temporal_df(n=N, pos="MID"):
    """Simulated sorted (season, GW) training data."""
    seasons = ["2022-23"] * (n // 2) + ["2023-24"] * (n // 2)
    gws = list(range(1, n // 2 + 1)) * 2
    return pd.DataFrame({
        "season": seasons,
        "GW": gws,
        "f0": RNG.random(n), "f1": RNG.random(n),
        "f2": RNG.random(n), "f3": RNG.random(n), "f4": RNG.random(n),
        "total_points": RNG.integers(0, 12, n).astype(float),
    })

DF = _make_temporal_df()
X_TRAIN = DF[["f0", "f1", "f2", "f3", "f4"]]
Y_TRAIN = DF["total_points"]
FEAT_COLS = ["f0", "f1", "f2", "f3", "f4"]


# validate_training_data tests

def test_validate_passes_good_data():
    validate_training_data(DF, FEAT_COLS, pos="MID", min_rows=100)  # no exception


def test_validate_raises_insufficient_rows():
    with pytest.raises(ValueError, match="insufficient rows"):
        validate_training_data(DF.head(50), FEAT_COLS, pos="MID", min_rows=100)


def test_validate_raises_all_zero_feature():
    bad = DF.copy()
    bad["f0"] = 0.0
    with pytest.raises(ValueError, match="all-zero"):
        validate_training_data(bad, FEAT_COLS, pos="MID", min_rows=100)


def test_validate_raises_on_nan_rho_risk():
    """If target is constant, warn (don't raise — degenerate but not fatal)."""
    const = DF.copy()
    const["total_points"] = 5.0
    # Should not raise — promote.py handles NaN ρ gracefully
    validate_training_data(const, FEAT_COLS, pos="MID", min_rows=100)


# tune_position_model tests

def test_returns_model_and_metadata():
    model, algo, params, cv_rho = tune_position_model(
        pos="MID", X_train=X_TRAIN, y_train=Y_TRAIN,
        feat_cols=FEAT_COLS, algos=["rf", "xgb"], n_trials=2
    )
    assert algo in ("rf", "xgb")
    assert isinstance(params, dict)
    assert isinstance(cv_rho, float)
    assert hasattr(model, "predict")


def test_single_algo_rf():
    _, algo, _, _ = tune_position_model(
        pos="GK", X_train=X_TRAIN, y_train=Y_TRAIN,
        feat_cols=FEAT_COLS, algos=["rf"], n_trials=2
    )
    assert algo == "rf"


def test_single_algo_xgb():
    _, algo, _, _ = tune_position_model(
        pos="FWD", X_train=X_TRAIN, y_train=Y_TRAIN,
        feat_cols=FEAT_COLS, algos=["xgb"], n_trials=2
    )
    assert algo == "xgb"


def test_model_can_predict():
    model, _, _, _ = tune_position_model(
        pos="DEF", X_train=X_TRAIN, y_train=Y_TRAIN,
        feat_cols=FEAT_COLS, algos=["rf", "xgb"], n_trials=2
    )
    preds = model.predict(X_TRAIN)
    assert len(preds) == len(X_TRAIN)


def test_rf_has_oob_score():
    """RF model must be fit with oob_score=True so promote.py gets OOB metrics."""
    model, algo, _, _ = tune_position_model(
        pos="MID", X_train=X_TRAIN, y_train=Y_TRAIN,
        feat_cols=FEAT_COLS, algos=["rf"], n_trials=2
    )
    if algo == "rf":
        assert hasattr(model, "oob_score_"), "RF must have oob_score_ attribute"
```

- [ ] **Step 1.4: Run to confirm failure**

```bash
python -m pytest tests/test_tune.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.pipeline.tune'`

- [ ] **Step 1.5: Create `src/pipeline/tune.py`**

```python
# src/pipeline/tune.py
"""Optuna-based hyperparameter tuning for per-position RF and XGBoost models.

CV uses TimeSeriesSplit on data sorted by (season, GW) to prevent temporal leakage.
"""
import logging
import warnings

import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

_N_CV_FOLDS = 5
_VALID_ALGOS = frozenset({"rf", "xgb"})


def validate_training_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    pos: str,
    min_rows: int = 200,
) -> None:
    """Raise ValueError if training data is degenerate.

    Checks: minimum row count, no fully-zero feature columns.
    Logs a warning (does not raise) if target is constant — NaN ρ is
    handled gracefully by promote.py's benchmark comparison.
    """
    if len(df) < min_rows:
        raise ValueError(
            f"[tune] {pos}: insufficient rows ({len(df)} < {min_rows}). "
            "Run retrain after more GW data is available."
        )
    for col in feature_cols:
        if col in df.columns and (df[col].fillna(0) == 0).all():
            raise ValueError(
                f"[tune] {pos}: feature '{col}' is all-zero — "
                "likely a missing data source. Fix feature engineering before retraining."
            )
    if "total_points" in df.columns and df["total_points"].nunique() <= 1:
        logger.warning("[tune] %s: target is constant — CV ρ will be NaN", pos)


def _cv_rho_timeseries(model_fn, X: pd.DataFrame, y: pd.Series) -> float:
    """Mean Spearman ρ across TimeSeriesSplit folds.

    Data MUST be sorted by (season, GW) before calling — caller is responsible.
    TimeSeriesSplit respects temporal order: train always precedes val in time.
    """
    tss = TimeSeriesSplit(n_splits=_N_CV_FOLDS)
    rhos = []
    for train_idx, val_idx in tss.split(X):
        m = model_fn()
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = m.predict(X.iloc[val_idx])
        rho, _ = spearmanr(preds, y.iloc[val_idx])
        if not np.isnan(float(rho)):
            rhos.append(float(rho))
    return float(np.mean(rhos)) if rhos else float("nan")


def _rf_objective(X: pd.DataFrame, y: pd.Series):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
        return _cv_rho_timeseries(
            lambda: RandomForestRegressor(**params, random_state=42, n_jobs=-1), X, y
        )
    return objective


def _xgb_objective(X: pd.DataFrame, y: pd.Series):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        return _cv_rho_timeseries(
            lambda: XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0), X, y
        )
    return objective


def tune_position_model(
    pos: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feat_cols: list[str],
    algos: list[str] | None = None,
    n_trials: int = 50,
) -> tuple:
    """Tune RF and/or XGBoost for one position using Optuna TPE + TimeSeriesSplit CV.

    X_train must be sorted by (season, GW) — caller is responsible.

    Returns:
        (best_model, best_algo, best_params, best_cv_rho)
    """
    if algos is None:
        algos = ["rf", "xgb"]
    unknown = set(algos) - _VALID_ALGOS
    if unknown:
        raise ValueError(f"Unknown algo(s): {unknown}. Valid: {_VALID_ALGOS}")

    best_rho = float("-inf")
    best_model = None
    best_algo = None
    best_params: dict = {}

    for algo in algos:
        print(f"[tune] {pos}/{algo.upper()} — {n_trials} trials")
        objective = _rf_objective(X_train, y_train) if algo == "rf" else _xgb_objective(X_train, y_train)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        trial_rho = study.best_value
        trial_params = study.best_params
        print(f"[tune] {pos}/{algo.upper()} best CV ρ={trial_rho:.4f} params={trial_params}")

        if not np.isnan(trial_rho) and trial_rho > best_rho:
            best_rho = trial_rho
            best_algo = algo
            best_params = trial_params

            # Refit on full training set with best params
            if algo == "rf":
                best_model = RandomForestRegressor(
                    **trial_params, oob_score=True, random_state=42, n_jobs=-1
                )
            else:
                best_model = XGBRegressor(
                    **trial_params, random_state=42, n_jobs=-1, verbosity=0
                )
            best_model.fit(X_train[feat_cols], y_train)

    return best_model, best_algo, best_params, best_rho
```

- [ ] **Step 1.6: Run tests**

```bash
python -m pytest tests/test_tune.py -v
```
Expected: all tests PASS.

- [ ] **Step 1.7: Commit**

```bash
git add requirements.txt src/pipeline/tune.py tests/test_tune.py
git commit -m "feat(track-c): add tune.py — Optuna HPO with TimeSeriesSplit CV, data validation"
```

---

## Task 2: Wire `tune.py` into `phase_retrain` + live-ρ guard

**Files:**
- Modify: `src/pipeline/run.py`
- Modify: `tests/test_run.py`

- [ ] **Step 2.1: Write failing tests**

```python
# tests/test_run.py — add:
import pandas as pd
import numpy as np


def _make_retrain_df(n=500):
    rng = np.random.default_rng(0)
    seasons = ["2022-23"] * (n // 2) + ["2023-24"] * (n // 2)
    gws = list(range(1, n // 2 + 1)) * 2
    return pd.DataFrame({
        "season": seasons,
        "GW": gws,
        "position": ["GK"] * 125 + ["DEF"] * 125 + ["MID"] * 125 + ["FWD"] * 125,
        "total_points": rng.integers(0, 12, n).astype(float),
        **{f"f{i}": rng.random(n) for i in range(5)},
    })


def test_phase_retrain_calls_tune_position_model(monkeypatch):
    """phase_retrain must delegate to tune_position_model, not raw RF fit."""
    import src.pipeline.run as run_mod
    calls = []

    def fake_tune(pos, X_train, y_train, feat_cols, algos, n_trials):
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=5, random_state=0, oob_score=True)
        m.fit(X_train, y_train)
        calls.append(pos)
        return m, "rf", {"n_estimators": 5}, 0.5

    monkeypatch.setattr("src.pipeline.run.tune_position_model", fake_tune)
    monkeypatch.setattr("src.pipeline.run.validate_training_data", lambda *a, **kw: None)
    monkeypatch.setattr("src.pipeline.run.run_promotion_pipeline", lambda **kw: {})
    monkeypatch.setattr("src.pipeline.run.build_merged_dataset", lambda **kw: _make_retrain_df())
    monkeypatch.setattr("src.pipeline.run.engineer_features", lambda df: df)

    run_mod.phase_retrain(n_trials=2)
    assert set(calls) == {"GK", "DEF", "MID", "FWD"}, f"Expected all 4 positions, got {calls}"


def test_phase_retrain_blocked_by_live_rho_guard(monkeypatch, tmp_path):
    """phase_retrain must skip promotion when last 3 live ρ values are all negative."""
    import src.pipeline.run as run_mod
    import src.config as cfg

    # Write an accuracy_log with 3 consecutive negative ρ rows
    log_path = tmp_path / "accuracy_log.csv"
    log_path.write_text(
        "gw,season,spearman_rho\n"
        "31,2025-26,-0.15\n"
        "32,2025-26,-0.19\n"
        "33,2025-26,-0.04\n"
    )
    monkeypatch.setattr(cfg, "ACCURACY_LOG_PATH", log_path)

    promotion_calls = []
    monkeypatch.setattr("src.pipeline.run.tune_position_model",
        lambda **kw: (__import__('sklearn.ensemble', fromlist=['RandomForestRegressor'])
                      .RandomForestRegressor(n_estimators=5).fit([[0]]*10, [0]*10),
                      "rf", {}, 0.5))
    monkeypatch.setattr("src.pipeline.run.validate_training_data", lambda *a, **kw: None)
    monkeypatch.setattr("src.pipeline.run.run_promotion_pipeline",
                        lambda **kw: promotion_calls.append(1))
    monkeypatch.setattr("src.pipeline.run.build_merged_dataset", lambda **kw: _make_retrain_df())
    monkeypatch.setattr("src.pipeline.run.engineer_features", lambda df: df)

    run_mod.phase_retrain(n_trials=2)
    assert len(promotion_calls) == 0, "Promotion must be blocked when live ρ < 0 for 3 GWs"
```

- [ ] **Step 2.2: Run to confirm failures**

```bash
python -m pytest tests/test_run.py::test_phase_retrain_calls_tune_position_model \
               tests/test_run.py::test_phase_retrain_blocked_by_live_rho_guard -v
```

- [ ] **Step 2.3: Add `ACCURACY_LOG_PATH` to `src/config.py`**

```python
# In config.py, after RESULTS_DIR definition:
ACCURACY_LOG_PATH = RESULTS_DIR / "accuracy_log.csv"
```

- [ ] **Step 2.4: Rewrite `phase_retrain` in `run.py`**

```python
def phase_retrain(
    target_gw: int | None = None,
    n_trials: int = 50,
    algos: list[str] | None = None,
):
    """Phase 4: Retrain 4 per-position models with Optuna HPO (RF vs XGB).

    Skips promotion if the last 3 completed GWs in accuracy_log.csv all have
    Spearman ρ < 0 — a signal of distribution shift requiring investigation.
    """
    from datetime import date as _date
    import joblib
    from src.pipeline.tune import tune_position_model, validate_training_data
    from src.pipeline.promote import run_promotion_pipeline
    from src.config import (
        BENCHMARK_PATH, METRICS_LEDGER_PATH, CHARTS_DIR,
        CURRENT_SEASON, ACCURACY_LOG_PATH,
    )

    if algos is None:
        algos = ["rf", "xgb"]

    # Live-ρ guard: block promotion when model is systematically failing on live data
    _check_live_rho_guard(ACCURACY_LOG_PATH)

    print("[retrain] Building full feature-engineered dataset...")
    merged = build_merged_dataset(vaastav_dir=VAASTAV_DIR)
    features = engineer_features(merged)
    print(f"[retrain] Training data: {len(features)} rows")

    if "position" in features.columns and pd.api.types.is_integer_dtype(features["position"]):
        features["position"] = features["position"].map(ELEMENT_TYPE_MAP)

    feature_cols = [c for c in ALL_FEATURE_COLUMNS if c in features.columns]
    date_str = _date.today().strftime("%Y%m%d")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    position_results = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_df = features[features["position"] == pos].copy()

        # Sort by (season, GW) — required for TimeSeriesSplit in tune.py
        if "season" in pos_df.columns and "GW" in pos_df.columns:
            pos_df = pos_df.sort_values(["season", "GW"]).reset_index(drop=True)

        try:
            validate_training_data(pos_df, feature_cols, pos=pos, min_rows=200)
        except ValueError as e:
            print(f"[retrain] {pos}: skipped — {e}")
            continue

        X = pos_df[feature_cols].fillna(0)
        y = pos_df["total_points"]

        model, algo, params, cv_rho = tune_position_model(
            pos=pos, X_train=X, y_train=y,
            feat_cols=feature_cols, algos=algos, n_trials=n_trials,
        )

        new_path = MODELS_DIR / f"{algo}_{pos.lower()}_{date_str}.sav"
        joblib.dump(model, new_path)
        position_results[pos] = {
            "rho": cv_rho, "path": new_path,
            "model": model, "algo": algo, "params": params,
        }
        print(f"[retrain] {pos}: {algo.upper()} CV-ρ={cv_rho:.4f} → {new_path.name}")

    if not position_results:
        print("[retrain] No positions trained. Check data quality.")
        return

    print("\n[retrain] Running promotion pipeline...")
    run_promotion_pipeline(
        trained_models={pos: (r["model"], r["path"]) for pos, r in position_results.items()},
        algorithm=None,   # mixed per-position — promote.py reads algo from path stem
        features_df=features,
        feature_cols=feature_cols,
        date_str=date_str,
        model_dir=MODELS_DIR,
        benchmark_path=BENCHMARK_PATH,
        ledger_path=METRICS_LEDGER_PATH,
        charts_dir=CHARTS_DIR,
        current_season=CURRENT_SEASON,
    )


def _check_live_rho_guard(log_path) -> None:
    """Raise RuntimeError if last 3 GWs in accuracy_log all have ρ < 0.

    This guards against promoting a model that is systematically failing on
    live data — even if it passes the walk-forward benchmark gate.
    """
    from pathlib import Path
    p = Path(log_path)
    if not p.exists():
        return
    try:
        log = pd.read_csv(p)
    except Exception:
        return
    if "spearman_rho" not in log.columns:
        return
    recent = log["spearman_rho"].dropna().tail(3)
    if len(recent) >= 3 and (recent < 0).all():
        raise RuntimeError(
            f"[retrain] BLOCKED: last {len(recent)} GWs all have live ρ < 0 "
            f"({recent.tolist()}). Investigate distribution shift before retraining. "
            "Override by passing --skip-rho-guard."
        )
```

- [ ] **Step 2.5: Add `--n-trials`, `--algos`, `--skip-rho-guard` to `main()` argparse**

```python
parser.add_argument("--n-trials", type=int, default=50,
                    help="Optuna trials per algo per position (retrain only)")
parser.add_argument("--algos", nargs="+", default=["rf", "xgb"],
                    choices=["rf", "xgb"])
parser.add_argument("--skip-rho-guard", action="store_true",
                    help="Skip live-ρ guard and allow promotion even with negative live ρ")

# In retrain dispatch:
elif args.phase == "retrain":
    if args.skip_rho_guard:
        import unittest.mock
        with unittest.mock.patch("src.pipeline.run._check_live_rho_guard"):
            phase_retrain(args.gw, n_trials=args.n_trials, algos=args.algos)
    else:
        phase_retrain(args.gw, n_trials=args.n_trials, algos=args.algos)
```

- [ ] **Step 2.6: Run tests**

```bash
python -m pytest tests/test_run.py -v
```

- [ ] **Step 2.7: Commit (DO NOT merge to master until Task 3 is also complete)**

```bash
git add src/pipeline/run.py src/config.py tests/test_run.py
git commit -m "feat(track-c): wire Optuna HPO + live-ρ guard into phase_retrain"
```

---

## Task 3: Fix `promote.py` — per-position algo in ledger & manifest

**⚠️ Must be completed in the same deploy as Task 2.** Task 2 passes `algorithm=None` to `run_promotion_pipeline`; without this fix, the ledger writes `algorithm: null` and the manifest breaks CI model downloads.

**Files:**
- Modify: `src/pipeline/promote.py`
- Modify: `tests/test_promote.py`

- [ ] **Step 3.1: Write failing test**

```python
# tests/test_promote.py — add:
def test_promote_records_per_position_algo(tmp_path):
    """Ledger must record the actual algo inferred from filename, not shared string."""
    from src.pipeline.promote import run_promotion_pipeline
    import pandas as pd, numpy as np, joblib
    from xgboost import XGBRegressor

    rng = np.random.default_rng(0)
    n = 200
    features_df = pd.DataFrame({
        "season": ["2024-25"] * n,
        "GW": list(range(1, n + 1)),
        "position": ["MID"] * n,
        "total_points": rng.integers(0, 12, n).astype(float),
        "f0": rng.random(n), "f1": rng.random(n),
    })
    feat_cols = ["f0", "f1"]
    xgb = XGBRegressor(n_estimators=5, random_state=0, verbosity=0)
    xgb.fit(features_df[feat_cols], features_df["total_points"])
    xgb_path = tmp_path / "xgb_mid_20260419.sav"
    joblib.dump(xgb, xgb_path)

    run_promotion_pipeline(
        trained_models={"MID": (xgb, xgb_path)},
        algorithm=None,
        features_df=features_df,
        feature_cols=feat_cols,
        date_str="20260419",
        model_dir=tmp_path,
        benchmark_path=tmp_path / "benchmark.json",
        ledger_path=tmp_path / "metrics_history.jsonl",
        charts_dir=tmp_path / "charts",
        current_season="2024-25",
    )

    import json
    record = json.loads((tmp_path / "metrics_history.jsonl").read_text().strip().splitlines()[-1])
    assert record.get("algorithm") == "xgb", f"Expected 'xgb', got {record.get('algorithm')!r}"


def test_promote_raises_on_unknown_algo_in_filename(tmp_path):
    """Inferred algo from filename must be in {'rf', 'xgb'} — raise if not."""
    from src.pipeline.promote import run_promotion_pipeline
    import pandas as pd, numpy as np, joblib
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    n = 100
    features_df = pd.DataFrame({
        "season": ["2024-25"] * n,
        "GW": list(range(1, n + 1)),
        "position": ["GK"] * n,
        "total_points": rng.integers(0, 12, n).astype(float),
        "f0": rng.random(n),
    })
    m = RandomForestRegressor(n_estimators=5, random_state=0)
    m.fit(features_df[["f0"]], features_df["total_points"])
    bad_path = tmp_path / "catboost_gk_20260419.sav"  # unknown algo
    joblib.dump(m, bad_path)

    with pytest.raises(ValueError, match="unknown algo"):
        run_promotion_pipeline(
            trained_models={"GK": (m, bad_path)},
            algorithm=None,
            features_df=features_df,
            feature_cols=["f0"],
            date_str="20260419",
            model_dir=tmp_path,
            benchmark_path=tmp_path / "benchmark.json",
            ledger_path=tmp_path / "metrics_history.jsonl",
            charts_dir=tmp_path / "charts",
            current_season="2024-25",
        )
```

- [ ] **Step 3.2: Run to confirm failures**

```bash
python -m pytest tests/test_promote.py::test_promote_records_per_position_algo \
               tests/test_promote.py::test_promote_raises_on_unknown_algo_in_filename -v
```

- [ ] **Step 3.3: Read `promote.py` to find where `algorithm` is written to ledger**

```bash
grep -n "algorithm" src/pipeline/promote.py
```

Find the line in `run_promotion_pipeline` where `result_entry` (or equivalent dict) sets `"algorithm": algorithm`. This is the line to patch.

- [ ] **Step 3.4: Apply the fix in `promote.py`**

At the line where `algorithm` is stored in the ledger record dict, replace:

```python
# Before:
"algorithm": algorithm,

# After:
"algorithm": _infer_algo(algorithm, path),
```

Add helper at module top:

```python
_VALID_ALGOS = frozenset({"rf", "xgb"})

def _infer_algo(algorithm: str | None, path) -> str:
    """Return algo name: use provided value or infer from path stem first segment."""
    if algorithm is not None:
        return algorithm
    inferred = Path(path).stem.split("_")[0]
    if inferred not in _VALID_ALGOS:
        raise ValueError(
            f"unknown algo '{inferred}' inferred from {path.name}. "
            f"Filename must start with one of {_VALID_ALGOS}."
        )
    return inferred
```

Also update `build_active_models_manifest`: where it reads `algorithm` from the shared arg, replace with per-position algo from the ledger/promoted dict (already in `result_entry["algorithm"]` after the fix above — no further change needed if `build_active_models_manifest` reads from `result_entry`).

- [ ] **Step 3.5: Run all promote tests**

```bash
python -m pytest tests/test_promote.py -v
```
Expected: all pass (new + existing).

- [ ] **Step 3.6: Commit Tasks 2+3 together**

```bash
git add src/pipeline/promote.py tests/test_promote.py
git commit -m "fix(promote): per-position algo inferred from filename; validate against known algos"
```

---

## Task 4: SHAP explainability — `compute_shap_reasons`

**Files:**
- Modify: `src/pipeline/predict.py`
- Create: `tests/test_shap_explain.py`

- [ ] **Step 4.1: Write failing tests**

```python
# tests/test_shap_explain.py
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.pipeline.predict import compute_shap_reasons

RNG = np.random.default_rng(0)
N = 100
FEAT_COLS = ["minutes_roll_4", "ict_index_roll_4", "xGC_rolling_4", "is_home", "transfers_net"]
X = pd.DataFrame(RNG.random((N, len(FEAT_COLS))), columns=FEAT_COLS)
Y = pd.Series(RNG.integers(0, 12, N).astype(float))


@pytest.fixture
def rf_model():
    m = RandomForestRegressor(n_estimators=10, random_state=0)
    m.fit(X, Y)
    return m


@pytest.fixture
def xgb_model():
    m = XGBRegressor(n_estimators=10, random_state=0, verbosity=0)
    m.fit(X, Y)
    return m


def test_returns_series_same_length(rf_model):
    result = compute_shap_reasons(rf_model, X, FEAT_COLS, top_n=3)
    assert len(result) == N


def test_reason_string_format(rf_model):
    result = compute_shap_reasons(rf_model, X, FEAT_COLS, top_n=3)
    sample = result.iloc[0]
    assert isinstance(sample, str)
    parts = [p.strip() for p in sample.split("|")]
    assert len(parts) == 3
    for part in parts:
        assert ":" in part
        # Sign prefix present
        value_str = part.split(":")[1].strip()
        assert value_str.startswith("+") or value_str.startswith("-")


def test_works_with_xgb(xgb_model):
    result = compute_shap_reasons(xgb_model, X, FEAT_COLS, top_n=2)
    assert len(result) == N
    assert result.iloc[0].count("|") == 1  # 2 parts → 1 pipe


def test_top_n_respected(rf_model):
    for top_n in [1, 2, 3]:
        result = compute_shap_reasons(rf_model, X, FEAT_COLS, top_n=top_n)
        parts = result.iloc[0].split("|")
        assert len(parts) == top_n


def test_raises_on_column_mismatch(rf_model):
    """Feature column mismatch must raise, not silently produce wrong SHAP labels."""
    wrong_cols = ["col_a", "col_b", "col_c", "col_d", "col_e"]
    X_wrong = pd.DataFrame(RNG.random((N, 5)), columns=wrong_cols)
    with pytest.raises(ValueError, match="column mismatch"):
        compute_shap_reasons(rf_model, X_wrong, wrong_cols, top_n=3)
```

- [ ] **Step 4.2: Run to confirm failure**

```bash
python -m pytest tests/test_shap_explain.py -v
```

- [ ] **Step 4.3: Implement `compute_shap_reasons` in `predict.py`**

Add import at top of file:

```python
import shap
```

Add function:

```python
_VALID_ALGOS_FOR_SHAP = (
    "sklearn.ensemble._forest.RandomForestRegressor",
    "xgboost.sklearn.XGBRegressor",
)


def compute_shap_reasons(
    model,
    X: pd.DataFrame,
    feature_cols: list[str],
    top_n: int = 3,
) -> pd.Series:
    """Return pipe-separated top-N SHAP feature contributions per player row.

    Format: "minutes_roll_4: +2.14 | ict_index_roll_4: +1.03 | is_home: -0.41"
    Positive = drove xP up, negative = drove xP down.

    Raises ValueError on feature column mismatch to prevent silent wrong labels.
    """
    # Guard: column alignment check
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        if expected != feature_cols:
            raise ValueError(
                f"compute_shap_reasons column mismatch: "
                f"model expects {expected[:5]}..., got {feature_cols[:5]}..."
            )

    X_clean = X[feature_cols].fillna(0)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_clean)  # shape (n, p)

    reasons = []
    for row_shap in shap_values:
        abs_idx = np.argsort(np.abs(row_shap))[::-1][:top_n]
        parts = [
            f"{feature_cols[i]}: {'+' if row_shap[i] >= 0 else ''}{row_shap[i]:.2f}"
            for i in abs_idx
        ]
        reasons.append(" | ".join(parts))
    return pd.Series(reasons, index=X.index)
```

- [ ] **Step 4.4: Run tests**

```bash
python -m pytest tests/test_shap_explain.py -v
```
Expected: all PASS (note: `test_raises_on_column_mismatch` passes only if the model has `feature_names_in_` — XGBRegressor does, sklearn RF does since sklearn ≥1.0).

- [ ] **Step 4.5: Commit**

```bash
git add src/pipeline/predict.py tests/test_shap_explain.py
git commit -m "feat(track-c): add compute_shap_reasons() with column mismatch guard"
```

---

## Task 5: Attach `shap_reason` to predictions & recommend outputs

**Files:**
- Modify: `src/pipeline/predict.py` — call `compute_shap_reasons` in `predict_next_gw_per_position`; add `"shap_reason": "first"` to DGW aggregation
- Modify: `src/pipeline/recommend.py` — propagate `shap_reason` into recommend output and `recommend.csv`
- Modify: `tests/test_predict_position.py`
- Modify: `tests/test_recommend.py`

- [ ] **Step 5.1: Read `predict_next_gw_per_position` to find the DGW aggregation block**

```bash
grep -n "groupby\|agg_cols\|shap" src/pipeline/predict.py
```

Note the line numbers of the groupby/agg block — you'll add `"shap_reason": "first"` there.

- [ ] **Step 5.2: Write failing tests**

```python
# tests/test_predict_position.py — add:
def test_predictions_have_shap_reason(tmp_path, monkeypatch):
    """predict_next_gw_per_position output must include shap_reason column."""
    # Re-use an existing fixture from this file that creates player_features + model_paths
    # (adapt variable names to match what's already in the file)
    result = predict_next_gw_per_position(player_features, model_paths)
    assert "shap_reason" in result.columns, "shap_reason column missing"
    assert result["shap_reason"].notna().all(), "shap_reason has NaN values"
    sample = result["shap_reason"].iloc[0]
    assert "|" in sample, f"Expected pipe-separated reasons, got: {sample!r}"


def test_dgw_shap_reason_preserved(tmp_path):
    """DGW players (2 fixture rows) must retain shap_reason after aggregation."""
    # Build a two-fixture player scenario and verify shap_reason survives groupby
    result = predict_next_gw_per_position(dgw_player_features, model_paths)
    dgw_player = result[result["element"] == DGW_ELEMENT_ID]
    assert len(dgw_player) == 1, "DGW player should be de-duplicated to one row"
    assert dgw_player["shap_reason"].iloc[0] != "", "shap_reason must be non-empty after DGW aggregation"
```

- [ ] **Step 5.3: Run to confirm failures**

```bash
python -m pytest tests/test_predict_position.py::test_predictions_have_shap_reason \
               tests/test_predict_position.py::test_dgw_shap_reason_preserved -v
```

- [ ] **Step 5.4: Update `predict_next_gw_per_position` in `predict.py`**

After `pos_df["xP"] = model.predict(X)` add:

```python
pos_df["shap_reason"] = compute_shap_reasons(model, X, feature_cols, top_n=3)
```

In the DGW aggregation `agg_cols` dict, add:

```python
"shap_reason": "first",  # fixture 1 SHAP reasons used for DGW; fixture 2 dropped (known limitation)
```

- [ ] **Step 5.5: Write failing recommend test**

```python
# tests/test_recommend.py — add:
def test_recommend_output_includes_shap_reason(predictions_df, user_state):
    """Transfer recommendation must include non-empty shap_reason for incoming player."""
    predictions_df = predictions_df.copy()
    predictions_df["shap_reason"] = "minutes_roll_4: +2.1 | ict_index_roll_4: +1.0 | is_home: +0.5"

    plan = recommend_transfers(predictions_df, user_state, horizon=1)

    assert len(plan["transfers"]) > 0, "Test fixture must produce at least one transfer"
    for transfer in plan["transfers"]:
        assert "shap_reason" in transfer, "Transfer entry missing shap_reason"
        assert transfer["shap_reason"] != "", "shap_reason must be non-empty"
```

- [ ] **Step 5.6: Run to confirm failure**

```bash
python -m pytest tests/test_recommend.py::test_recommend_output_includes_shap_reason -v
```

- [ ] **Step 5.7: Update `recommend.py`** — attach `shap_reason` when building transfer entry dict, and write it to `recommend.csv` via `save_recommend_csv`.

- [ ] **Step 5.8: Run all predict + recommend tests**

```bash
python -m pytest tests/test_predict_position.py tests/test_recommend.py -v
```

- [ ] **Step 5.9: Commit**

```bash
git add src/pipeline/predict.py src/pipeline/recommend.py \
        tests/test_predict_position.py tests/test_recommend.py
git commit -m "feat(track-c): shap_reason in predictions.csv and recommend.csv; DGW aggregation handled"
```

---

## Task 6: E2E smoke test + docs

**Files:**
- Modify: `tests/test_pipeline_e2e.py`
- Modify: `docs/improvements-roadmap.md`

- [ ] **Step 6.1: Write E2E smoke test**

```python
# tests/test_pipeline_e2e.py — add:
def test_retrain_hpo_smoke(monkeypatch, tmp_path):
    """phase_retrain with n_trials=2 + XGB only completes and writes .sav files."""
    import src.pipeline.run as run_mod
    import src.config as cfg
    import pandas as pd

    # Provide enough GWs so live-ρ guard does not block (only 2 rows, guard needs ≥3)
    log = tmp_path / "accuracy_log.csv"
    log.write_text("gw,season,spearman_rho\n31,2025-26,0.45\n32,2025-26,0.38\n")
    monkeypatch.setattr(cfg, "ACCURACY_LOG_PATH", log)
    monkeypatch.setattr(run_mod, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(run_mod, "run_promotion_pipeline", lambda **kw: None)
    monkeypatch.setattr(run_mod, "build_merged_dataset", lambda **kw: _make_e2e_df())
    monkeypatch.setattr(run_mod, "engineer_features", lambda df: df)

    run_mod.phase_retrain(n_trials=2, algos=["xgb"])

    sav_files = list(tmp_path.glob("*.sav"))
    assert len(sav_files) == 4, f"Expected 4 .sav files (one per position), got: {[f.name for f in sav_files]}"
    for f in sav_files:
        assert f.stem.startswith("xgb_"), f"Expected xgb_ prefix, got {f.stem}"


def _make_e2e_df(n=500):
    import numpy as np
    rng = np.random.default_rng(1)
    import pandas as pd
    seasons = ["2022-23"] * (n // 2) + ["2023-24"] * (n // 2)
    gws = list(range(1, n // 2 + 1)) * 2
    return pd.DataFrame({
        "season": seasons,
        "GW": gws,
        "position": ["GK"] * 125 + ["DEF"] * 125 + ["MID"] * 125 + ["FWD"] * 125,
        "total_points": rng.integers(0, 12, n).astype(float),
        **{f"f{i}": rng.random(n) for i in range(6)},
    })
```

- [ ] **Step 6.2: Run full test suite**

```bash
python -m pytest tests/ -q
```
Expected: all existing + new tests PASS.

- [ ] **Step 6.3: Update `docs/improvements-roadmap.md`** — change Track C status to `✅ COMPLETE (2026-04-19)`, add built items summary.

- [ ] **Step 6.4: Final commit**

```bash
git add tests/test_pipeline_e2e.py docs/improvements-roadmap.md
git commit -m "test(track-c): E2E HPO smoke test; mark Track C complete"
```

---

## Usage after implementation

```bash
# Default: RF + XGB, 50 trials per algo per position (~20-30 min on CPU)
python -m src.pipeline.run retrain --gw 34

# Quick validation (2 trials, XGB only — ~2 min)
python -m src.pipeline.run retrain --gw 34 --n-trials 2 --algos xgb

# Override live-ρ guard (e.g. investigating distribution shift)
python -m src.pipeline.run retrain --gw 34 --skip-rho-guard

# predictions.csv now includes:
# shap_reason: "ict_index_roll_4: +2.14 | is_home: +1.03 | minutes_roll_4: +0.87"

# recommend.csv transfer entries now include shap_reason for incoming player
```

## Known limitations

- **DGW SHAP**: For double-gameweek players, SHAP reasons reflect fixture 1 only. Fixture 2 reasons are dropped during aggregation. Accepted limitation; label as such in output.
- **HPO runtime**: 50 trials × 5 folds × 4 positions × 2 algos ≈ 2000 model fits. Expect 20–40 min on CPU. Monthly retrain cadence makes this acceptable.
- **Live-ρ guard**: Blocks promotion when last 3 GWs have ρ < 0. The current GW32-33 values (-0.19, -0.04) would trigger this — investigate distribution shift (target leakage, feature availability differences between vaastav and live FPL API) before running retrain.

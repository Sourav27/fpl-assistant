# Track I — Model Registry & Automated Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Agent type:** This plan is optimised for the **mlops-engineer** subagent. All code is Python within the `src/pipeline/` production package.

**Goal:** Replace manual model promotion with a fully automated pipeline: walk-forward evaluation on the current season, per-position benchmark comparison, PNG chart generation, and GitHub Release publishing with a manifest that CI reads at runtime.

**Architecture:** A new `src/pipeline/promote.py` module handles all promotion logic — evaluation, benchmark comparison, ledger writes, chart generation, and release publishing. `phase_retrain()` in `run.py` calls it after training. CI (`daily_bootstrap.yml`) downloads `active_models.json` from the latest GitHub Release and downloads only the model files listed in it.

**Tech Stack:** Python 3.11, pandas, scikit-learn, scipy, matplotlib, joblib, subprocess (`gh` CLI), pytest.

---

## Prerequisite context for the agent

This is the `fpl-assistant` FPL ML pipeline. Key facts:

- **Working directory:** `D:\FPL\fpl-assistant`. Shell is bash on Windows. Use `rtk` prefix for all bash commands.
- **Current promoted models:** `models/rf_gk_20260412.sav`, `models/rf_def_20260412.sav`, `models/rf_mid_20260412.sav`, `models/rf_fwd_20260412.sav` (Track B GW31 retrain).
- **Current benchmark:** `models/benchmark_gw31.json` — will be superseded by `models/benchmark.json`.
- **Seasons:** vaastav data covers 2016-17 → 2024-25. Current season 2025-26 is patched from FPL API. `CURRENT_SEASON = "2025-26"` in `src/config.py`.
- **Feature columns:** `ALL_FEATURE_COLUMNS` (24 features) from `src/pipeline/predict.py`. Position strings: `"GK"`, `"DEF"`, `"MID"`, `"FWD"`.
- **Hauler definition:** players scoring ≥ 5 actual `total_points` in a GW.
- **`build_merged_dataset()`** in `prepare.py` returns a DataFrame with columns including `season`, `GW`, `position`, `total_points`, and all feature columns.
- **`engineer_features()`** in `features.py` takes the merged DataFrame and returns a feature-engineered DataFrame.
- **`ELEMENT_TYPE_MAP`** in `fetch.py`: `{1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}`.
- **260 tests currently passing** — do not break them.
- **Run tests with:** `python -m pytest tests/ -q`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/pipeline/promote.py` | **Create** | All promotion logic: evaluation, benchmark, ledger, charts, release |
| `src/config.py` | Modify | Add `BENCHMARK_PATH`, `METRICS_LEDGER_PATH`, `CHARTS_DIR`, `get_active_models()` |
| `src/pipeline/run.py` | Modify | `phase_retrain()` calls `run_promotion_pipeline()` after training; `oob_score=True` |
| `.github/workflows/daily_bootstrap.yml` | Modify | Read `active_models.json` manifest; download listed `.sav` files |
| `scripts/download_models_from_manifest.py` | **Create** | Standalone script: download `.sav` files listed in manifest from a given release tag |
| `tests/test_promote.py` | **Create** | All tests for promote.py |
| `models/benchmark.json` | **Create** | Per-position best-ever metrics (replaces `benchmark_gw31.json`) |
| `.gitignore` | Modify | Add `!models/benchmark.json` and `!models/metrics_history.jsonl` exclusions |
| `docs/CLAUDE.md` | Modify | Update model promotion section |

---

## Data Schemas

### `models/benchmark.json`
```json
{
  "GK":  {"model_file": "rf_gk_20260412.sav",  "algorithm": "rf", "date": "2026-04-12", "train_mae": 0.45, "train_rho": 0.85, "train_hauler_mae": 3.2, "test_mae": 0.724, "test_rho": 0.742, "test_hauler_mae": 4.1, "test_gw_count": 8},
  "DEF": {"model_file": "rf_def_20260412.sav",  "algorithm": "rf", "date": "2026-04-12", "train_mae": 0.80, "train_rho": 0.72, "train_hauler_mae": 4.5, "test_mae": 1.189, "test_rho": 0.613, "test_hauler_mae": 5.8, "test_gw_count": 8},
  "MID": {"model_file": "rf_mid_20260412.sav",  "algorithm": "rf", "date": "2026-04-12", "train_mae": 0.70, "train_rho": 0.79, "train_hauler_mae": 4.0, "test_mae": 1.085, "test_rho": 0.717, "test_hauler_mae": 5.2, "test_gw_count": 8},
  "FWD": {"model_file": "rf_fwd_20260412.sav",  "algorithm": "rf", "date": "2026-04-12", "train_mae": 0.75, "train_rho": 0.78, "train_hauler_mae": 4.3, "test_mae": 1.266, "test_rho": 0.721, "test_hauler_mae": 5.6, "test_gw_count": 8}
}
```
> **Note:** train metrics above are placeholders — Task 2 recomputes them from the actual GW31 models.

### `models/metrics_history.jsonl` (one JSON object per line)
```json
{"run_id": "20260412_143022", "date": "2026-04-12", "algorithm": "rf", "position": "GK", "model_file": "rf_gk_20260412.sav", "training_rows": 13065, "feature_columns": ["total_points_roll_4", "..."], "train_mae": 0.45, "train_rho": 0.85, "train_hauler_mae": 3.2, "test_mae": 0.724, "test_rho": 0.742, "test_hauler_mae": 4.1, "test_gw_count": 8, "per_gw_metrics": [{"gw": 1, "mae": 0.71, "rho": 0.74, "hauler_mae": 4.0}], "promoted": true, "benchmark_test_rho_at_time": 0.700}
```

### `active_models.json` (GitHub Release asset)
```json
{
  "published": "2026-04-12",
  "models": {
    "GK":  {"file": "rf_gk_20260412.sav",  "algorithm": "rf", "date": "2026-04-12", "test_rho": 0.742, "test_mae": 0.724},
    "DEF": {"file": "rf_def_20260412.sav", "algorithm": "rf", "date": "2026-04-12", "test_rho": 0.613, "test_mae": 1.189},
    "MID": {"file": "rf_mid_20260412.sav", "algorithm": "rf", "date": "2026-04-12", "test_rho": 0.717, "test_mae": 1.085},
    "FWD": {"file": "rf_fwd_20260412.sav", "algorithm": "rf", "date": "2026-04-12", "test_rho": 0.721, "test_mae": 1.266}
  }
}
```

---

## Task 1: Write test shells (failing tests — TDD first)

**Files:**
- Create: `tests/test_promote.py`

Do NOT implement any production code in this task. Write all tests first; run them to confirm they fail for the right reason (`ImportError` or `AttributeError`).

- [ ] **Step 1: Create `tests/test_promote.py` with the following content**

```python
# tests/test_promote.py
"""Tests for src/pipeline/promote.py — model registry and automated promotion (Track I)."""
import json
import math
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_features():
    """Minimal feature DataFrame: 2 positions × 5 GWs × 3 players = 30 rows.
    Includes a current-season split (season="2025-26") and a training split.
    """
    rows = []
    for season, gws in [("2024-25", range(1, 6)), ("2025-26", range(1, 6))]:
        for gw in gws:
            for code, pos in [(1, "GK"), (2, "MID")]:
                rows.append({
                    "code": code, "element": code, "season": season, "GW": gw,
                    "position": pos, "total_points": 4 + code,
                    **{f"feat_{i}": float(i + gw) for i in range(6)},
                })
    return pd.DataFrame(rows)


@pytest.fixture
def dummy_model():
    m = MagicMock()
    m.predict.return_value = [3.5, 4.0, 3.0]
    m.feature_importances_ = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]
    m.feature_names_in_ = [f"feat_{i}" for i in range(6)]
    return m


@pytest.fixture
def sample_benchmark(tmp_path):
    bench = {
        "GK":  {"model_file": "rf_gk_20260101.sav", "algorithm": "rf", "date": "2026-01-01",
                "train_mae": 0.5, "train_rho": 0.80, "train_hauler_mae": 3.5,
                "test_mae": 0.80, "test_rho": 0.70, "test_hauler_mae": 4.5, "test_gw_count": 5},
        "MID": {"model_file": "rf_mid_20260101.sav", "algorithm": "rf", "date": "2026-01-01",
                "train_mae": 0.8, "train_rho": 0.75, "train_hauler_mae": 4.0,
                "test_mae": 1.10, "test_rho": 0.71, "test_hauler_mae": 5.2, "test_gw_count": 5},
    }
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(bench))
    return path


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_mae_computed_correctly(self):
        from src.pipeline.promote import compute_metrics
        df = pd.DataFrame({"y_pred": [2.0, 4.0, 6.0], "y_actual": [1.0, 4.0, 5.0]})
        result = compute_metrics(df)
        assert result["mae"] == pytest.approx(2 / 3, abs=0.01)

    def test_rho_perfect_rank_order(self):
        from src.pipeline.promote import compute_metrics
        df = pd.DataFrame({"y_pred": [1.0, 2.0, 3.0], "y_actual": [2.0, 4.0, 6.0]})
        result = compute_metrics(df)
        assert result["rho"] == pytest.approx(1.0, abs=0.01)

    def test_hauler_mae_only_includes_gte5_pts(self):
        from src.pipeline.promote import compute_metrics
        df = pd.DataFrame({"y_pred": [3.0, 6.0, 8.0], "y_actual": [2.0, 5.0, 10.0]})
        result = compute_metrics(df)
        # Only rows where y_actual >= 5: preds [6.0, 8.0], actuals [5.0, 10.0]
        # hauler_mae = (|6-5| + |8-10|) / 2 = 1.5
        assert result["hauler_mae"] == pytest.approx(1.5, abs=0.01)

    def test_hauler_mae_nan_when_no_haulers(self):
        from src.pipeline.promote import compute_metrics
        df = pd.DataFrame({"y_pred": [1.0, 2.0], "y_actual": [1.0, 2.0]})
        result = compute_metrics(df)
        assert math.isnan(result["hauler_mae"])

    def test_returns_nan_rho_for_empty_df(self):
        from src.pipeline.promote import compute_metrics
        result = compute_metrics(pd.DataFrame({"y_pred": [], "y_actual": []}))
        assert math.isnan(result["rho"])


# ---------------------------------------------------------------------------
# evaluate_current_season
# ---------------------------------------------------------------------------

class TestEvaluateCurrentSeason:
    def test_returns_per_gw_metrics_for_test_season(self, sample_features, dummy_model):
        from src.pipeline.promote import evaluate_current_season
        feat_cols = [f"feat_{i}" for i in range(6)]
        result = evaluate_current_season(
            model=dummy_model,
            position="GK",
            features_df=sample_features,
            feature_cols=feat_cols,
            current_season="2025-26",
        )
        assert "per_gw" in result
        assert len(result["per_gw"]) > 0
        assert all("gw" in g and "mae" in g and "rho" in g and "hauler_mae" in g
                   for g in result["per_gw"])

    def test_returns_aggregated_test_metrics(self, sample_features, dummy_model):
        from src.pipeline.promote import evaluate_current_season
        feat_cols = [f"feat_{i}" for i in range(6)]
        result = evaluate_current_season(
            model=dummy_model, position="GK",
            features_df=sample_features, feature_cols=feat_cols,
            current_season="2025-26",
        )
        assert "test_mae" in result
        assert "test_rho" in result
        assert "test_hauler_mae" in result
        assert "test_gw_count" in result
        assert "test_gw_elapsed" in result  # DS-7: track actual GWs elapsed

    def test_returns_train_metrics(self, sample_features, dummy_model):
        from src.pipeline.promote import evaluate_current_season
        feat_cols = [f"feat_{i}" for i in range(6)]
        result = evaluate_current_season(
            model=dummy_model, position="GK",
            features_df=sample_features, feature_cols=feat_cols,
            current_season="2025-26",
        )
        assert "train_mae" in result
        assert "train_rho" in result
        assert "train_hauler_mae" in result

    def test_no_data_leakage_train_excludes_current_season(self, sample_features, dummy_model):
        """Train rows passed to model.predict must be from historical seasons only.

        We remove oob_prediction_ so the function falls back to model.predict()
        for train metrics, making the call count observable.
        """
        from src.pipeline.promote import evaluate_current_season
        feat_cols = [f"feat_{i}" for i in range(6)]
        # Remove oob_prediction_ to force predict()-based train evaluation
        dummy_model.oob_prediction_ = None  # length mismatch → falls back to predict
        call_rows = []
        def capture_predict(X):
            call_rows.append(len(X))
            return [3.0] * len(X)
        dummy_model.predict.side_effect = capture_predict
        evaluate_current_season(
            model=dummy_model, position="GK",
            features_df=sample_features, feature_cols=feat_cols,
            current_season="2025-26",
        )
        # First predict call = train (2024-25 GKs only = 5 rows)
        # Second predict call = pooled test (2025-26 GKs = 5 rows)
        assert call_rows[0] == 5  # 1 GK × 5 GWs in 2024-25 only


# ---------------------------------------------------------------------------
# load_benchmark / compare_to_benchmark / update_benchmark
# ---------------------------------------------------------------------------

class TestBenchmarkOperations:
    def test_load_benchmark_returns_dict(self, sample_benchmark):
        from src.pipeline.promote import load_benchmark
        bench = load_benchmark(sample_benchmark)
        assert "GK" in bench
        assert bench["GK"]["test_rho"] == pytest.approx(0.70, abs=0.01)

    def test_load_benchmark_returns_empty_dict_when_missing(self, tmp_path):
        from src.pipeline.promote import load_benchmark
        bench = load_benchmark(tmp_path / "nonexistent.json")
        assert bench == {}

    def test_compare_promotes_when_rho_improves(self, sample_benchmark):
        from src.pipeline.promote import load_benchmark, compare_to_benchmark
        bench = load_benchmark(sample_benchmark)
        # rho improves (0.75 > 0.70) and MAE does not degrade (0.78 <= 0.80 * 1.05)
        new_metrics = {"GK": {"test_rho": 0.75, "test_mae": 0.78, "test_gw_elapsed": 10}}
        result = compare_to_benchmark(new_metrics, bench, min_test_gws=6)
        assert result["GK"] is True

    def test_compare_skips_when_rho_does_not_improve(self, sample_benchmark):
        from src.pipeline.promote import load_benchmark, compare_to_benchmark
        bench = load_benchmark(sample_benchmark)
        new_metrics = {"GK": {"test_rho": 0.65, "test_mae": 0.70, "test_gw_elapsed": 10}}
        result = compare_to_benchmark(new_metrics, bench, min_test_gws=6)
        assert result["GK"] is False

    def test_compare_skips_when_mae_degrades_significantly(self, sample_benchmark):
        from src.pipeline.promote import load_benchmark, compare_to_benchmark
        bench = load_benchmark(sample_benchmark)
        # rho improves but MAE is >5% worse than benchmark (0.80)
        new_metrics = {"GK": {"test_rho": 0.76, "test_mae": 0.90, "test_gw_elapsed": 10}}
        result = compare_to_benchmark(new_metrics, bench, min_test_gws=6)
        assert result["GK"] is False

    def test_compare_skips_when_too_few_gws(self, sample_benchmark):
        from src.pipeline.promote import load_benchmark, compare_to_benchmark
        bench = load_benchmark(sample_benchmark)
        # rho improves but only 3 GWs elapsed (below min_test_gws=6)
        new_metrics = {"GK": {"test_rho": 0.80, "test_mae": 0.70, "test_gw_elapsed": 3}}
        result = compare_to_benchmark(new_metrics, bench, min_test_gws=6)
        assert result["GK"] is False

    def test_compare_promotes_when_no_benchmark_exists_for_position(self):
        from src.pipeline.promote import compare_to_benchmark
        # No benchmark for FWD → promote freely (first-ever model for this position)
        result = compare_to_benchmark(
            {"FWD": {"test_rho": 0.60, "test_mae": 1.30, "test_gw_elapsed": 10}},
            benchmark={},
            min_test_gws=6,
        )
        assert result["FWD"] is True

    def test_update_benchmark_writes_new_entry(self, tmp_path):
        from src.pipeline.promote import update_benchmark
        path = tmp_path / "benchmark.json"
        metrics = {
            "test_rho": 0.76, "test_mae": 0.71, "test_hauler_mae": 4.0,
            "test_gw_count": 8, "train_mae": 0.44, "train_rho": 0.86,
            "train_hauler_mae": 3.1, "model_file": "rf_gk_20260501.sav",
            "algorithm": "rf", "date": "2026-05-01",
        }
        update_benchmark(path, position="GK", metrics=metrics, existing={})
        saved = json.loads(path.read_text())
        assert saved["GK"]["test_rho"] == pytest.approx(0.76, abs=0.01)
        assert saved["GK"]["model_file"] == "rf_gk_20260501.sav"

    def test_update_benchmark_preserves_other_positions(self, sample_benchmark):
        from src.pipeline.promote import load_benchmark, update_benchmark
        existing = load_benchmark(sample_benchmark)
        metrics = {
            "test_rho": 0.76, "test_mae": 0.71, "test_hauler_mae": 4.0,
            "test_gw_count": 8, "train_mae": 0.44, "train_rho": 0.86,
            "train_hauler_mae": 3.1, "model_file": "rf_gk_20260501.sav",
            "algorithm": "rf", "date": "2026-05-01",
        }
        update_benchmark(sample_benchmark, position="GK", metrics=metrics, existing=existing)
        saved = json.loads(sample_benchmark.read_text())
        assert "MID" in saved  # preserved


# ---------------------------------------------------------------------------
# append_metrics_ledger
# ---------------------------------------------------------------------------

class TestMetricsLedger:
    def test_creates_file_on_first_write(self, tmp_path):
        from src.pipeline.promote import append_metrics_ledger
        path = tmp_path / "metrics_history.jsonl"
        record = {"run_id": "test", "position": "GK", "test_rho": 0.74}
        append_metrics_ledger(path, record)
        assert path.exists()

    def test_appends_on_second_write(self, tmp_path):
        from src.pipeline.promote import append_metrics_ledger
        path = tmp_path / "metrics_history.jsonl"
        append_metrics_ledger(path, {"run_id": "r1", "position": "GK"})
        append_metrics_ledger(path, {"run_id": "r2", "position": "DEF"})
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_each_line_is_valid_json(self, tmp_path):
        from src.pipeline.promote import append_metrics_ledger
        path = tmp_path / "metrics_history.jsonl"
        append_metrics_ledger(path, {"run_id": "r1", "position": "GK", "promoted": True})
        record = json.loads(path.read_text().strip())
        assert record["promoted"] is True


# ---------------------------------------------------------------------------
# build_active_models_manifest
# ---------------------------------------------------------------------------

class TestBuildManifest:
    def test_manifest_has_all_four_positions(self):
        from src.pipeline.promote import build_active_models_manifest
        promoted = {
            "GK":  {"model_file": "rf_gk_20260412.sav", "algorithm": "rf", "date": "2026-04-12", "test_rho": 0.742, "test_mae": 0.724},
            "DEF": {"model_file": "rf_def_20260412.sav", "algorithm": "rf", "date": "2026-04-12", "test_rho": 0.613, "test_mae": 1.189},
            "MID": {"model_file": "rf_mid_20260412.sav", "algorithm": "rf", "date": "2026-04-12", "test_rho": 0.717, "test_mae": 1.085},
            "FWD": {"model_file": "rf_fwd_20260412.sav", "algorithm": "rf", "date": "2026-04-12", "test_rho": 0.721, "test_mae": 1.266},
        }
        manifest = build_active_models_manifest(promoted, published_date="2026-04-12")
        assert set(manifest["models"].keys()) == {"GK", "DEF", "MID", "FWD"}
        assert manifest["models"]["GK"]["file"] == "rf_gk_20260412.sav"

    def test_manifest_includes_published_date(self):
        from src.pipeline.promote import build_active_models_manifest
        manifest = build_active_models_manifest({}, published_date="2026-04-12")
        assert manifest["published"] == "2026-04-12"


# ---------------------------------------------------------------------------
# generate_charts
# ---------------------------------------------------------------------------

class TestGenerateCharts:
    def test_produces_png_files(self, tmp_path, dummy_model):
        from src.pipeline.promote import generate_charts
        history = [
            {"date": "2026-01-01", "position": "GK", "test_rho": 0.70, "test_mae": 0.80, "per_gw_metrics": [{"gw": 1, "rho": 0.70}]},
            {"date": "2026-04-12", "position": "GK", "test_rho": 0.742, "test_mae": 0.724, "per_gw_metrics": [{"gw": 1, "rho": 0.742}]},
        ]
        models = {"GK": dummy_model}
        feat_cols = [f"feat_{i}" for i in range(6)]
        charts = generate_charts(history=history, models=models, feature_cols=feat_cols, output_dir=tmp_path)
        assert len(charts) > 0
        for path in charts:
            assert Path(path).suffix == ".png"
            assert Path(path).exists()
```

- [ ] **Step 2: Run new tests to confirm they fail for the right reason**

```bash
python -m pytest tests/test_promote.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'compute_metrics' from 'src.pipeline.promote'` (module doesn't exist yet).

- [ ] **Step 3: Confirm existing tests still pass**

```bash
python -m pytest tests/ -q --ignore=tests/test_promote.py 2>&1 | tail -5
```

Expected: `260 passed, 1 skipped`.

- [ ] **Step 4: Commit test shells**

```bash
rtk git add tests/test_promote.py && rtk git commit -m "test: Track I test shells for model registry and automated promotion"
```

---

## Task 2: Config updates + benchmark.json bootstrap

**Files:**
- Modify: `src/config.py`
- Create: `models/benchmark.json`

- [ ] **Step 1: Add constants to `src/config.py`**

Read `src/config.py` first. After the `SNAPSHOTS_DIR` line, add:

```python
# Model registry paths (Track I)
BENCHMARK_PATH      = MODELS_DIR / "benchmark.json"
METRICS_LEDGER_PATH = MODELS_DIR / "metrics_history.jsonl"
CHARTS_DIR          = MODELS_DIR / "charts"
```

- [ ] **Step 2: Create `models/benchmark.json`**

This seeds the benchmark from the GW31 retrain results. Train metrics were not persisted at GW31 retrain time, so they are omitted (null) and will be populated on the next retrain run.

```json
{
  "GK":  {"model_file": "rf_gk_20260412.sav",  "algorithm": "rf", "date": "2026-04-12", "train_mae": null, "train_rho": null, "train_hauler_mae": null, "test_mae": 0.724, "test_rho": 0.742, "test_hauler_mae": null, "test_gw_count": null},
  "DEF": {"model_file": "rf_def_20260412.sav",  "algorithm": "rf", "date": "2026-04-12", "train_mae": null, "train_rho": null, "train_hauler_mae": null, "test_mae": 1.189, "test_rho": 0.613, "test_hauler_mae": null, "test_gw_count": null},
  "MID": {"model_file": "rf_mid_20260412.sav",  "algorithm": "rf", "date": "2026-04-12", "train_mae": null, "train_rho": null, "train_hauler_mae": null, "test_mae": 1.085, "test_rho": 0.717, "test_hauler_mae": null, "test_gw_count": null},
  "FWD": {"model_file": "rf_fwd_20260412.sav",  "algorithm": "rf", "date": "2026-04-12", "train_mae": null, "train_rho": null, "train_hauler_mae": null, "test_mae": 1.266, "test_rho": 0.721, "test_hauler_mae": null, "test_gw_count": null}
}
```

- [ ] **Step 3: Verify and update `.gitignore`**

`models/` is git-ignored for `.sav` files. `benchmark.json` and `metrics_history.jsonl` must be tracked. Check:

```bash
rtk git check-ignore -v models/benchmark.json models/metrics_history.jsonl
```

If either is matched, open `.gitignore` and add explicit exclusions immediately after the `models/` line:

```gitignore
models/
!models/benchmark.json
!models/metrics_history.jsonl
```

Verify the exclusion works before committing:

```bash
rtk git status models/benchmark.json
```

Expected: file appears as untracked (not ignored).

- [ ] **Step 5: Add config tests**

In `tests/test_config.py`, add:

```python
def test_benchmark_path_is_in_models_dir():
    from src.config import BENCHMARK_PATH, MODELS_DIR
    assert BENCHMARK_PATH.parent == MODELS_DIR

def test_metrics_ledger_path_is_in_models_dir():
    from src.config import METRICS_LEDGER_PATH, MODELS_DIR
    assert METRICS_LEDGER_PATH.parent == MODELS_DIR

def test_charts_dir_is_in_models_dir():
    from src.config import CHARTS_DIR, MODELS_DIR
    assert CHARTS_DIR.parent == MODELS_DIR
```

- [ ] **Step 6: Run config tests**

```bash
python -m pytest tests/test_config.py -v 2>&1 | tail -10
```

Expected: new tests pass.

- [ ] **Step 7: Commit**

```bash
rtk git add src/config.py models/benchmark.json tests/test_config.py .gitignore && rtk git commit -m "feat: I-F0 add benchmark.json seed and Track I config paths"
```

---

## Task 3: Core metrics functions (`promote.py` — Part 1)

**Files:**
- Create: `src/pipeline/promote.py`
- Test: `tests/test_promote.py`

Implement `compute_metrics`, `evaluate_current_season`, and the benchmark operations.

- [ ] **Step 1: Create `src/pipeline/promote.py` with `compute_metrics`**

```python
# src/pipeline/promote.py
"""Model registry and automated promotion pipeline (Track I).

Public API:
  compute_metrics(df, hauler_threshold)                 -> dict
  evaluate_current_season(model, position, features_df,
                          feature_cols, current_season) -> dict
  load_benchmark(path)                                  -> dict
  compare_to_benchmark(new_metrics, benchmark,
                       min_test_gws)                    -> dict[str, bool]
  update_benchmark(path, position, metrics, existing)   -> None
  append_metrics_ledger(path, record)                   -> None
  build_active_models_manifest(promoted, published_date) -> dict
  generate_charts(history, models, feature_cols,
                  output_dir)                           -> list[str]
  run_promotion_pipeline(trained_models, algorithm,
                         features_df, feature_cols,
                         date_str, model_dir)           -> dict
"""
import json
import math
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# Position-specific hauler thresholds (total_points >= threshold = hauler).
# GK/DEF: clean sheet + bonus = ~9 pts. MID/FWD: one goal = 4-5 pts (too low); use 9/11.
HAULER_THRESHOLDS: dict[str, int] = {"GK": 7, "DEF": 9, "MID": 9, "FWD": 11}
_DEFAULT_HAULER_THRESHOLD = 9  # fallback when position is not passed


def compute_metrics(df: pd.DataFrame, hauler_threshold: int = _DEFAULT_HAULER_THRESHOLD) -> dict:
    """Compute MAE, Spearman rho, and hauler MAE from a prediction DataFrame.

    Args:
        df: DataFrame with columns y_pred and y_actual.
        hauler_threshold: total_points value at or above which a player is a "hauler".
                          Use HAULER_THRESHOLDS[position] for position-specific values.

    Returns:
        dict with keys: mae, rho, hauler_mae (float, NaN when undefined).
    """
    df = df.dropna(subset=["y_pred", "y_actual"])
    if len(df) < 2:
        return {"mae": float("nan"), "rho": float("nan"), "hauler_mae": float("nan")}

    mae = float(np.mean(np.abs(df["y_pred"] - df["y_actual"])))

    rho_val, _ = spearmanr(df["y_pred"], df["y_actual"])
    rho = float(rho_val) if not math.isnan(float(rho_val)) else float("nan")

    haulers = df[df["y_actual"] >= hauler_threshold]
    if len(haulers) >= 2:
        hauler_mae = float(np.mean(np.abs(haulers["y_pred"] - haulers["y_actual"])))
    else:
        hauler_mae = float("nan")

    return {"mae": mae, "rho": rho, "hauler_mae": hauler_mae}
```

- [ ] **Step 2: Run compute_metrics tests**

```bash
python -m pytest tests/test_promote.py::TestComputeMetrics -v
```

Expected: all 5 tests pass.

- [ ] **Step 3: Add `evaluate_current_season` to `promote.py`**

```python
def evaluate_current_season(
    model,
    position: str,
    features_df: pd.DataFrame,
    feature_cols: list[str],
    current_season: str,
) -> dict:
    """Evaluate a trained model: OOB train metrics on historical data, pooled test
    metrics on the current season.

    NOTE: This is NOT a walk-forward (per-GW retrain). The model was trained once on
    all historical seasons. "walk-forward" here means per-GW evaluation for diagnostic
    charts only; the promotion-driving test_rho/test_mae are computed on the full
    pooled current-season predictions to avoid averaging correlated correlation
    coefficients.

    Train split: all rows where season != current_season AND position == position.
                 OOB predictions are used when model.oob_prediction_ is available
                 (requires oob_score=True at training time) to avoid in-sample bias.
    Test split: rows where season == current_season AND position == position,
                all GWs pooled into a single prediction set.

    Returns dict with keys:
        train_mae, train_rho, train_hauler_mae,
        test_mae, test_rho, test_hauler_mae, test_gw_count, test_gw_elapsed,
        per_gw: list of {gw, mae, rho, hauler_mae}  (diagnostic only, not used for promotion)
    """
    hauler_thresh = HAULER_THRESHOLDS.get(position, _DEFAULT_HAULER_THRESHOLD)
    pos_df = features_df[features_df["position"] == position].copy()
    available_cols = [c for c in feature_cols if c in pos_df.columns]

    # --- Train metrics (OOB to avoid in-sample RF bias) ---
    train_df = pos_df[pos_df["season"] != current_season].copy()
    if len(train_df) >= 2:
        if hasattr(model, "oob_prediction_") and len(model.oob_prediction_) == len(train_df):
            # OOB predictions are out-of-sample (no bias from RF memorising train data)
            y_train_pred = model.oob_prediction_
        else:
            # Fallback: in-sample (optimistic but acceptable if OOB unavailable)
            logger.warning(
                f"[promote] {position}: OOB predictions unavailable — "
                "train metrics are in-sample (optimistic). Enable oob_score=True "
                "in RandomForestRegressor for honest train metrics."
            )
            X_train = train_df[available_cols].fillna(0)
            y_train_pred = model.predict(X_train)
        train_cmp = pd.DataFrame({"y_pred": y_train_pred, "y_actual": train_df["total_points"].values})
        train_metrics = compute_metrics(train_cmp, hauler_threshold=hauler_thresh)
    else:
        train_metrics = {"mae": float("nan"), "rho": float("nan"), "hauler_mae": float("nan")}

    # --- Test metrics: pooled rho/MAE on the full current season ---
    test_df = pos_df[pos_df["season"] == current_season].copy()
    test_gw_elapsed = int(test_df["GW"].nunique())  # actual GWs in season

    if len(test_df) >= 2:
        X_all = test_df[available_cols].fillna(0)
        y_pred_all = model.predict(X_all)
        pooled_cmp = pd.DataFrame({"y_pred": y_pred_all, "y_actual": test_df["total_points"].values})
        pooled_metrics = compute_metrics(pooled_cmp, hauler_threshold=hauler_thresh)
        test_agg = {
            "test_mae": pooled_metrics["mae"],
            "test_rho": pooled_metrics["rho"],
            "test_hauler_mae": pooled_metrics["hauler_mae"],
            "test_gw_count": test_gw_elapsed,
            "test_gw_elapsed": test_gw_elapsed,
        }
    else:
        test_agg = {
            "test_mae": float("nan"), "test_rho": float("nan"),
            "test_hauler_mae": float("nan"), "test_gw_count": 0,
            "test_gw_elapsed": test_gw_elapsed,
        }

    # --- Per-GW diagnostic metrics (for charts only — not used for promotion) ---
    per_gw = []
    for gw, gw_df in test_df.groupby("GW"):
        if len(gw_df) < 2:
            continue
        X_gw = gw_df[available_cols].fillna(0)
        y_pred = model.predict(X_gw)
        cmp = pd.DataFrame({"y_pred": y_pred, "y_actual": gw_df["total_points"].values})
        m = compute_metrics(cmp, hauler_threshold=hauler_thresh)
        per_gw.append({"gw": int(gw), "mae": m["mae"], "rho": m["rho"], "hauler_mae": m["hauler_mae"]})

    return {
        "train_mae": train_metrics["mae"],
        "train_rho": train_metrics["rho"],
        "train_hauler_mae": train_metrics["hauler_mae"],
        **test_agg,
        "per_gw": per_gw,
    }
```

- [ ] **Step 4: Run `evaluate_current_season` tests**

```bash
python -m pytest tests/test_promote.py::TestEvaluateCurrentSeason -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Add benchmark operations to `promote.py`**

```python
def load_benchmark(path: Path) -> dict:
    """Load benchmark.json. Returns empty dict if file does not exist."""
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


MIN_TEST_GWS = 6  # require at least 6 completed GWs before any promotion decision


def compare_to_benchmark(
    new_metrics: dict,
    benchmark: dict,
    min_test_gws: int = MIN_TEST_GWS,
) -> dict:
    """Compare new per-position metrics against benchmark. Promotes only if rho
    improves AND MAE does not degrade significantly AND enough GWs have elapsed.

    Args:
        new_metrics: {position: {test_rho, test_mae, test_gw_elapsed, ...}}
        benchmark:   {position: {test_rho, test_mae, ...}}
        min_test_gws: minimum GWs elapsed in current season before any promotion.
                      Prevents noise-driven promotions at season start.

    Returns:
        {position: bool} — True = promote, False = skip.
    """
    decisions = {}
    for pos, metrics in new_metrics.items():
        new_rho = metrics.get("test_rho", float("nan"))
        new_mae = metrics.get("test_mae", float("nan"))
        gw_elapsed = metrics.get("test_gw_elapsed", metrics.get("test_gw_count", 0))

        # Guard: insufficient season data
        if gw_elapsed < min_test_gws:
            logger.info(
                f"[promote] {pos}: SKIPPED (only {gw_elapsed} GWs elapsed, "
                f"need {min_test_gws})"
            )
            decisions[pos] = False
            continue

        if math.isnan(new_rho):
            decisions[pos] = False
            continue

        if pos not in benchmark:
            decisions[pos] = True  # no prior benchmark → always promote
            continue

        bench_rho = benchmark[pos].get("test_rho", float("nan"))
        bench_mae = benchmark[pos].get("test_mae", float("nan"))

        rho_improves = not math.isnan(bench_rho) and (new_rho > bench_rho)
        if math.isnan(bench_rho):
            rho_improves = True  # no benchmark rho → promote freely

        # MAE non-regression: allow up to 5% slack on top of benchmark MAE
        if math.isnan(bench_mae) or math.isnan(new_mae):
            mae_ok = True
        else:
            mae_ok = new_mae <= bench_mae * 1.05

        decisions[pos] = rho_improves and mae_ok

    return decisions


def update_benchmark(path: Path, position: str, metrics: dict, existing: dict) -> None:
    """Update benchmark.json for a single position. Preserves all other positions."""
    path = Path(path)
    updated = dict(existing)
    updated[position] = metrics
    path.write_text(json.dumps(updated, indent=2))
```

- [ ] **Step 6: Run benchmark tests**

```bash
python -m pytest tests/test_promote.py::TestBenchmarkOperations -v
```

Expected: all 9 tests pass (7 original + 2 new: MAE degradation + min_test_gws guard).

- [ ] **Step 7: Run full test suite**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

- [ ] **Step 8: Commit**

```bash
rtk git add src/pipeline/promote.py && rtk git commit -m "feat: I-F1 compute_metrics, evaluate_current_season, benchmark operations"
```

---

## Task 4: Metrics ledger and manifest

**Files:**
- Modify: `src/pipeline/promote.py`
- Test: `tests/test_promote.py`

- [ ] **Step 1: Add `append_metrics_ledger` and `build_active_models_manifest` to `promote.py`**

```python
def append_metrics_ledger(path: Path, record: dict) -> None:
    """Append a single record (dict) as a JSON line to metrics_history.jsonl.

    Creates the file if it does not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def build_active_models_manifest(promoted: dict, published_date: str) -> dict:
    """Build the active_models.json manifest for a GitHub Release.

    Args:
        promoted: {position: {model_file, algorithm, date, test_rho, test_mae, ...}}
        published_date: ISO date string e.g. "2026-04-12"

    Returns:
        Manifest dict suitable for json.dumps.
    """
    models_block = {}
    for pos, info in promoted.items():
        models_block[pos] = {
            "file": info["model_file"],
            "algorithm": info["algorithm"],
            "date": info["date"],
            "test_rho": info.get("test_rho"),
            "test_mae": info.get("test_mae"),
        }
    return {"published": published_date, "models": models_block}
```

- [ ] **Step 2: Run ledger and manifest tests**

```bash
python -m pytest tests/test_promote.py::TestMetricsLedger tests/test_promote.py::TestBuildManifest -v
```

Expected: all 5 tests pass.

- [ ] **Step 3: Commit**

```bash
rtk git add src/pipeline/promote.py && rtk git commit -m "feat: I-F2 metrics ledger append and active_models manifest builder"
```

---

## Task 5: Chart generation

**Files:**
- Modify: `src/pipeline/promote.py`
- Test: `tests/test_promote.py`

Charts use `matplotlib`. It is already in `requirements.txt` (used elsewhere). Do NOT add it as a new dependency.

- [ ] **Step 1: Verify matplotlib is available**

```bash
python -c "import matplotlib; print(matplotlib.__version__)"
```

If missing, add `matplotlib>=3.7` to `requirements.txt`.

- [ ] **Step 2: Add `generate_charts` to `promote.py`**

Add to top-level imports: `import matplotlib; matplotlib.use("Agg")` (non-interactive backend for CI).

```python
def generate_charts(
    history: list[dict],
    models: dict,
    feature_cols: list[str],
    output_dir: Path,
) -> list[str]:
    """Generate PNG charts and save to output_dir.

    Charts produced:
      - metrics_rho_history.png   — test_rho over time, one line per position
      - metrics_mae_history.png   — test_mae over time, one line per position
      - metrics_per_gw_rho.png    — per-GW rho from the most recent run per position
      - feature_importance_{pos}.png — horizontal bar for each position with a model

    Args:
        history: list of ledger records (dicts with date, position, test_rho, test_mae,
                 per_gw_metrics fields). May span multiple positions and runs.
        models:  {position: fitted sklearn model} — for feature importance charts.
        feature_cols: list of feature names (in model training order).
        output_dir: directory to write PNG files into.

    Returns:
        List of absolute file path strings for all generated PNGs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    positions = ["GK", "DEF", "MID", "FWD"]
    colors = {"GK": "#2196F3", "DEF": "#4CAF50", "MID": "#FF9800", "FWD": "#F44336"}
    generated = []

    # --- rho history ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for pos in positions:
        pos_records = sorted(
            [r for r in history if r.get("position") == pos and r.get("test_rho") is not None],
            key=lambda r: r.get("date", ""),
        )
        if not pos_records:
            continue
        dates = [r["date"] for r in pos_records]
        rhos = [r["test_rho"] for r in pos_records]
        ax.plot(dates, rhos, marker="o", label=pos, color=colors[pos])
    ax.axhline(0.65, color="grey", linestyle="--", linewidth=0.8, label="target ρ=0.65")
    ax.set_title("Test Spearman ρ History (per-position)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean test Spearman ρ")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = output_dir / "metrics_rho_history.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    generated.append(str(path))

    # --- MAE history ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for pos in positions:
        pos_records = sorted(
            [r for r in history if r.get("position") == pos and r.get("test_mae") is not None],
            key=lambda r: r.get("date", ""),
        )
        if not pos_records:
            continue
        dates = [r["date"] for r in pos_records]
        maes = [r["test_mae"] for r in pos_records]
        ax.plot(dates, maes, marker="o", label=pos, color=colors[pos])
    ax.axhline(1.035, color="grey", linestyle="--", linewidth=0.8, label="baseline MAE=1.035")
    ax.set_title("Test MAE History (per-position)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean test MAE")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = output_dir / "metrics_mae_history.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    generated.append(str(path))

    # --- per-GW rho (most recent run per position) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for pos in positions:
        pos_records = sorted(
            [r for r in history if r.get("position") == pos],
            key=lambda r: r.get("date", ""),
        )
        if not pos_records:
            continue
        latest = pos_records[-1]
        per_gw = latest.get("per_gw_metrics") or latest.get("per_gw", [])
        if not per_gw:
            continue
        gws = [g["gw"] for g in per_gw]
        rhos = [g["rho"] for g in per_gw]
        ax.plot(gws, rhos, marker="o", label=pos, color=colors[pos])
    ax.axhline(0.65, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("Per-GW Test Spearman ρ (most recent run)")
    ax.set_xlabel("GW")
    ax.set_ylabel("Spearman ρ")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "metrics_per_gw_rho.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    generated.append(str(path))

    # --- feature importance per position ---
    for pos, model in models.items():
        if model is None:
            continue
        if hasattr(model, "feature_names_in_"):
            feat_names = list(model.feature_names_in_)
        else:
            feat_names = feature_cols[:len(model.feature_importances_)]
        importances = model.feature_importances_
        pairs = sorted(zip(importances, feat_names), reverse=True)[:15]
        imp_vals, feat_labels = zip(*pairs)
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(range(len(imp_vals)), imp_vals, color=colors.get(pos, "#999"))
        ax.set_yticks(range(len(feat_labels)))
        ax.set_yticklabels(feat_labels)
        ax.invert_yaxis()
        ax.set_title(f"Feature Importance — {pos}")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        path = output_dir / f"feature_importance_{pos.lower()}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        generated.append(str(path))

    return generated
```

- [ ] **Step 3: Run chart tests**

```bash
python -m pytest tests/test_promote.py::TestGenerateCharts -v
```

Expected: 1 test passes; PNG files are created.

- [ ] **Step 4: Run full suite**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
rtk git add src/pipeline/promote.py && rtk git commit -m "feat: I-F3 generate_charts (rho history, MAE history, per-GW rho, feature importance)"
```

---

## Task 6: `run_promotion_pipeline` orchestrator and GitHub Release

**Files:**
- Modify: `src/pipeline/promote.py`

This is the top-level function that `phase_retrain()` calls. It orchestrates the full flow and calls `gh release create`.

- [ ] **Step 1: Add `publish_release` and `run_promotion_pipeline` to `promote.py`**

```python
def publish_release(
    date_str: str,
    model_files: list[str],
    manifest: dict,
    chart_files: list[str],
    metrics_summary: dict,
    models_dir: Path,
    dry_run: bool = False,
) -> str:
    """Create a GitHub Release with models, manifest, and charts as assets.

    Release tag: model-{date_str} (e.g. model-20260412).
    Returns the release URL, or a dry-run note if dry_run=True.
    """
    import subprocess
    import tempfile

    tag = f"model-{date_str}"
    title = f"Models {date_str}"

    # Build notes table
    lines = ["## Model Promotion Report\n", f"Date: {date_str}\n\n",
             "| Position | Algorithm | test ρ | test MAE | Promoted |\n",
             "|----------|-----------|--------|----------|----------|\n"]
    for pos in ["GK", "DEF", "MID", "FWD"]:
        info = metrics_summary.get(pos, {})
        promoted = "✅" if info.get("promoted") else "⏭ kept"
        algo = info.get("algorithm", "?")
        rho = f"{info.get('test_rho', float('nan')):.3f}" if info.get('test_rho') is not None else "—"
        mae = f"{info.get('test_mae', float('nan')):.3f}" if info.get('test_mae') is not None else "—"
        lines.append(f"| {pos} | {algo} | {rho} | {mae} | {promoted} |\n")
    notes = "".join(lines)

    if dry_run:
        logger.info(f"[promote] DRY RUN — would create release {tag}")
        return f"dry-run:{tag}"

    # Idempotency: delete existing release with same tag before re-creating
    existing = subprocess.run(
        ["gh", "release", "view", tag], capture_output=True, text=True
    )
    if existing.returncode == 0:
        logger.info(f"[promote] Tag {tag} already exists — deleting before re-create")
        subprocess.run(["gh", "release", "delete", tag, "--yes"], check=True)

    manifest_path = None
    try:
        # Write manifest to a temp file (cleaned up in finally)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_active_models.json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(manifest, f, indent=2)
            manifest_path = f.name

        assets = [manifest_path] + model_files + chart_files
        cmd = ["gh", "release", "create", tag, *assets,
               "--title", title, "--notes", notes]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"[promote] gh release create failed (tag={tag}): {result.stderr.strip()}"
            )
        url = result.stdout.strip()
        logger.info(f"[promote] Release published: {url}")
        return url
    finally:
        if manifest_path:
            Path(manifest_path).unlink(missing_ok=True)


def run_promotion_pipeline(
    trained_models: dict,
    algorithm: str,
    features_df: pd.DataFrame,
    feature_cols: list[str],
    date_str: str | None = None,
    model_dir: Path | None = None,
    benchmark_path: Path | None = None,
    ledger_path: Path | None = None,
    charts_dir: Path | None = None,
    current_season: str = "2025-26",
    dry_run: bool = False,
) -> dict:
    """Full promotion pipeline. Call this after training position models.

    Args:
        trained_models: {position: (model, model_filepath)} where model_filepath
                        is the Path where the model was saved.
        algorithm:      e.g. "rf" or "xgb"
        features_df:    Full feature-engineered DataFrame (all seasons).
        feature_cols:   List of feature column names used in training.
        date_str:       YYYYMMDD string. Defaults to today.
        model_dir:      Path to models/ directory.
        benchmark_path: Path to benchmark.json.
        ledger_path:    Path to metrics_history.jsonl.
        charts_dir:     Path to save PNG charts.
        current_season: Season string for test split.
        dry_run:        If True, skip gh release create.

    Returns:
        {position: {promoted, test_rho, test_mae, model_file, ...}}
    """
    from src.config import (BENCHMARK_PATH, METRICS_LEDGER_PATH,
                             CHARTS_DIR, MODELS_DIR, CURRENT_SEASON)

    date_str = date_str or date.today().strftime("%Y%m%d")
    benchmark_path = benchmark_path or BENCHMARK_PATH
    ledger_path = ledger_path or METRICS_LEDGER_PATH
    charts_dir = charts_dir or CHARTS_DIR
    model_dir = model_dir or MODELS_DIR
    current_season = current_season or CURRENT_SEASON

    benchmark = load_benchmark(benchmark_path)
    run_id = f"{date_str}_{__import__('time').strftime('%H%M%S')}"

    # 1. Evaluate each position model
    eval_results = {}
    models_only = {}
    for pos, (model, model_path) in trained_models.items():
        if model is None:
            logger.warning(f"[promote] {pos}: no model object in trained_models — skipping")
            continue
        eval_res = evaluate_current_season(
            model=model,
            position=pos,
            features_df=features_df,
            feature_cols=feature_cols,
            current_season=current_season,
        )
        eval_results[pos] = eval_res
        models_only[pos] = model
        logger.info(
            f"[promote] {pos}: train_rho={eval_res['train_rho']:.3f}  "
            f"test_rho={eval_res['test_rho']:.3f}  "
            f"test_mae={eval_res['test_mae']:.3f}  "
            f"(test_gws={eval_res['test_gw_elapsed']})"
        )

    # 2. Compare to benchmark (includes min_test_gws + MAE non-regression gate)
    new_metrics_for_compare = {
        pos: {
            "test_rho": ev["test_rho"],
            "test_mae": ev["test_mae"],
            "test_gw_elapsed": ev.get("test_gw_elapsed", 0),
        }
        for pos, ev in eval_results.items()
    }
    promotion_decisions = compare_to_benchmark(new_metrics_for_compare, benchmark)

    # 3. Build full per-position result summary + metrics ledger
    summary = {}
    positions_to_promote: dict[str, dict] = {}  # collected before benchmark write
    promoted_models = {}
    model_files_to_upload = []

    for pos, (model, model_path) in trained_models.items():
        if pos not in eval_results:
            continue
        ev = eval_results[pos]
        model_file = Path(model_path).name
        promoted = promotion_decisions.get(pos, False)
        bench_rho = benchmark.get(pos, {}).get("test_rho", float("nan"))

        result_entry = {
            "model_file": model_file,
            "algorithm": algorithm,
            "date": date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:],
            "promoted": promoted,
            "test_rho": ev["test_rho"],
            "test_mae": ev["test_mae"],
            "test_hauler_mae": ev["test_hauler_mae"],
            "test_gw_count": ev["test_gw_count"],
            "test_gw_elapsed": ev.get("test_gw_elapsed", 0),
            "train_rho": ev["train_rho"],
            "train_mae": ev["train_mae"],
            "train_hauler_mae": ev["train_hauler_mae"],
            "benchmark_test_rho_at_time": bench_rho,
        }
        summary[pos] = result_entry

        # Append to metrics ledger
        ledger_record = {
            "run_id": run_id,
            "position": pos,
            "per_gw_metrics": ev["per_gw"],
            **result_entry,
        }
        append_metrics_ledger(ledger_path, ledger_record)

        if promoted:
            logger.info(f"[promote] {pos}: PROMOTED (rho {bench_rho:.3f} -> {ev['test_rho']:.3f})")
            promoted_models[pos] = result_entry
            model_files_to_upload.append(str(model_path))
            positions_to_promote[pos] = result_entry
        else:
            # Keep current benchmark model for this position in the manifest
            if pos in benchmark:
                promoted_models[pos] = {**benchmark[pos], "promoted": False}
                kept_path = model_dir / benchmark[pos]["model_file"]
                if not kept_path.exists():
                    raise FileNotFoundError(
                        f"[promote] Cannot publish release: benchmark model file "
                        f"'{kept_path}' missing for non-promoted position {pos}. "
                        f"Run retrain from a machine where all .sav files are present."
                    )
                model_files_to_upload.append(str(kept_path))
            logger.info(
                f"[promote] {pos}: SKIPPED — "
                f"rho={ev['test_rho']:.3f} bench_rho={bench_rho:.3f}"
            )

    # 4. Atomic benchmark update (single write after all decisions, not inside loop)
    if positions_to_promote:
        updated_benchmark = dict(load_benchmark(benchmark_path))
        for pos, entry in positions_to_promote.items():
            updated_benchmark[pos] = entry
        benchmark_path.write_text(json.dumps(updated_benchmark, indent=2))

    # 5. Generate charts
    ledger_records = []
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            if line.strip():
                ledger_records.append(json.loads(line))

    chart_files = generate_charts(
        history=ledger_records,
        models=models_only,
        feature_cols=feature_cols,
        output_dir=charts_dir,
    )

    # 6. Build manifest and publish release
    manifest = build_active_models_manifest(
        promoted_models,
        published_date=date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:],
    )
    release_url = publish_release(
        date_str=date_str,
        model_files=model_files_to_upload,
        manifest=manifest,
        chart_files=chart_files,
        metrics_summary=summary,
        models_dir=model_dir,
        dry_run=dry_run,
    )
    logger.info(f"[promote] Release: {release_url}")

    return summary
```

- [ ] **Step 2: Run full test suite**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: 260 passed, 1 skipped + new promote tests all passing.

- [ ] **Step 3: Commit**

```bash
rtk git add src/pipeline/promote.py && rtk git commit -m "feat: I-F4 run_promotion_pipeline orchestrator and publish_release"
```

---

## Task 7: Wire into `phase_retrain()`

**Files:**
- Modify: `src/pipeline/run.py`

- [ ] **Step 1: Read the current `phase_retrain()` in `src/pipeline/run.py`**

Find the function (search for `def phase_retrain`). Note where models are saved with `joblib.dump`.

- [ ] **Step 2: Update `phase_retrain()` imports and model naming**

At the top of the function, after the existing imports, add:

```python
from src.pipeline.promote import run_promotion_pipeline
from src.config import BENCHMARK_PATH, METRICS_LEDGER_PATH, CHARTS_DIR, CURRENT_SEASON
```

Change the model filename from `rf_{pos.lower()}_{label}.sav` to use date-based naming, and **enable `oob_score=True`** on the `RandomForestRegressor` so `evaluate_current_season` can use OOB predictions for honest train metrics (avoids in-sample RF bias):

```python
from datetime import date as _date
date_str = _date.today().strftime("%Y%m%d")
# Replace: new_path = MODELS_DIR / f"rf_{pos.lower()}_{label}.sav"
new_path = MODELS_DIR / f"{algorithm}_{pos.lower()}_{date_str}.sav"
```

When constructing the `RandomForestRegressor` (search for `RandomForestRegressor(` in `run.py`), add `oob_score=True`:

```python
# Before: RandomForestRegressor(n_estimators=100, ...)
# After:
model = RandomForestRegressor(n_estimators=100, ..., oob_score=True)
```

`oob_score=True` has negligible performance cost (uses the same trees already built) and makes the resulting `model.oob_prediction_` available for train-time evaluation without refitting.

where `algorithm = "rf"` (hardcoded for now; Track C will parameterise this).

- [ ] **Step 3: Collect trained models dict and call `run_promotion_pipeline`**

After the training loop, add:

```python
    if position_results:
        print("\n[retrain] Running promotion pipeline...")
        trained_models = {
            pos: (r["model"], r["path"])
            for pos, r in position_results.items()
            if "model" in r  # model object stored in result
        }
        run_promotion_pipeline(
            trained_models=trained_models,
            algorithm="rf",
            features_df=features,
            feature_cols=feature_cols,
            date_str=date_str,
            model_dir=MODELS_DIR,
            benchmark_path=BENCHMARK_PATH,
            ledger_path=METRICS_LEDGER_PATH,
            charts_dir=CHARTS_DIR,
            current_season=CURRENT_SEASON,
        )
```

Also update the result dict to store the model object:

```python
position_results[pos] = {"mae": mae, "rho": rho, "path": new_path, "n": len(pos_df), "model": model}
```

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

- [ ] **Step 5: Smoke test the retrain import chain**

```bash
python -c "from src.pipeline.run import phase_retrain; print('import ok')"
```

Expected: `import ok`

- [ ] **Step 6: Commit**

```bash
rtk git add src/pipeline/run.py && rtk git commit -m "feat: I-F5 wire run_promotion_pipeline into phase_retrain with date-based naming"
```

---

## Task 8: Update CI to read `active_models.json` manifest

**Files:**
- Modify: `.github/workflows/daily_bootstrap.yml`
- Create: `scripts/download_models_from_manifest.py`
- Modify: `src/config.py`

- [ ] **Step 1: Read the current model-download step in `.github/workflows/daily_bootstrap.yml`**

Find the step that runs `gh release list` and `gh release download`. Note the exact step name and commands.

- [ ] **Step 2: Create `scripts/download_models_from_manifest.py`**

The inline-Python-in-heredoc pattern used previously is broken (shell can't pass `$LATEST` as `sys.argv[1]` through a heredoc). Extract to a standalone script:

```python
#!/usr/bin/env python
# scripts/download_models_from_manifest.py
"""Download model files listed in models/active_models.json from a GitHub release.

Usage: python scripts/download_models_from_manifest.py <release_tag>
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: download_models_from_manifest.py <release_tag>", file=sys.stderr)
        sys.exit(1)

    tag = sys.argv[1]
    manifest_path = Path("models/active_models.json")
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    env = {**os.environ}  # GH_TOKEN must be set by caller

    failed = []
    for pos, info in manifest.get("models", {}).items():
        fname = info["file"]
        print(f"Downloading {pos}: {fname}")
        r = subprocess.run(
            ["gh", "release", "download", tag,
             "--pattern", fname, "--dir", "models", "--clobber"],
            capture_output=True, text=True, env=env,
        )
        if r.returncode != 0:
            print(f"  ERROR: {r.stderr.strip()}", file=sys.stderr)
            failed.append(fname)

    if failed:
        print(f"Failed to download: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Replace the CI model-download step**

Replace the existing model download step with:

```yaml
- name: Download models from latest release
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    # gh release list returns reverse-chronological order; take first match
    LATEST=$(gh release list --limit 20 --json tagName \
      --jq '[.[] | select(.tagName | test("^model-|^gw"))][0].tagName')
    echo "Latest release: $LATEST"
    mkdir -p models

    # Try to download active_models.json manifest
    if gh release download "$LATEST" --pattern "active_models.json" --dir models --clobber 2>/dev/null; then
      echo "Manifest found — downloading listed model files"
      python scripts/download_models_from_manifest.py "$LATEST"
    else
      echo "No manifest — falling back to downloading all *.sav files"
      gh release download "$LATEST" --pattern "*.sav" --dir models --clobber
    fi
    ls models/*.sav 2>/dev/null || echo "No .sav files downloaded"
```

- [ ] **Step 4: Update `src/config.py` — lazy manifest loading**

Running `_load_active_models_from_manifest()` at module import time creates test isolation problems (any `models/active_models.json` on the developer machine silently overrides `ACTIVE_MODELS`) and races with the CI download step. Use a lazy function instead:

In `src/config.py`, replace the module-level `_load_active_models_from_manifest` block with:

```python
def get_active_models() -> dict:
    """Return the active per-position model paths.

    Checks for models/active_models.json at call time (not import time).
    Falls back to the hardcoded ACTIVE_MODELS dict if the manifest is absent
    or malformed. Use this instead of referencing ACTIVE_MODELS directly in
    predict.py and run.py to pick up promotions without restarting.
    """
    import json as _json

    manifest_path = MODELS_DIR / "active_models.json"
    if manifest_path.exists():
        try:
            data = _json.loads(manifest_path.read_text())
            return {pos: MODELS_DIR / info["file"]
                    for pos, info in data.get("models", {}).items()}
        except Exception:
            pass
    return dict(ACTIVE_MODELS)
```

> **Note:** Callers in `predict.py` that reference `ACTIVE_MODELS` directly must be updated to call `get_active_models()` instead. Check `predict.py` for `ACTIVE_MODELS` references and replace.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

- [ ] **Step 6: Commit**

```bash
rtk git add .github/workflows/daily_bootstrap.yml scripts/download_models_from_manifest.py src/config.py && rtk git commit -m "feat: I-F6 CI reads active_models.json manifest; lazy get_active_models() in config"
```

---

## Task 9: Models folder cleanup

**Files:**
- No new production files

- [ ] **Step 1: Identify legacy model files to remove**

```bash
ls models/*.sav
```

Legacy files to remove (three generations that predate the GW31 named files):
- `models/rf_model.sav`
- `models/rf_model_gk.sav`, `rf_model_def.sav`, `rf_model_mid.sav`, `rf_model_fwd.sav`
- `models/xgb_model.sav`
- `models/xgb_model_gk.sav`, `xgb_model_def.sav`, `xgb_model_mid.sav`, `xgb_model_fwd.sav`
- `models/rf_model_gw31.sav` (monolithic, superseded by per-position)
- `models/benchmark_gw31.json` (superseded by `models/benchmark.json`)

Keep: `rf_gk_20260412.sav`, `rf_def_20260412.sav`, `rf_mid_20260412.sav`, `rf_fwd_20260412.sav`, `benchmark.json`, `metrics_history.jsonl`.

- [ ] **Step 2: Remove legacy files**

```bash
cd D:/FPL/fpl-assistant && rm models/rf_model.sav models/rf_model_gk.sav models/rf_model_def.sav models/rf_model_mid.sav models/rf_model_fwd.sav models/xgb_model.sav models/xgb_model_gk.sav models/xgb_model_def.sav models/xgb_model_mid.sav models/xgb_model_fwd.sav models/rf_model_gw31.sav models/benchmark_gw31.json 2>/dev/null; ls models/
```

- [ ] **Step 3: Verify models/ only contains current promoted files**

```bash
ls models/
```

Expected: `rf_gk_20260412.sav  rf_def_20260412.sav  rf_mid_20260412.sav  rf_fwd_20260412.sav  benchmark.json  metrics_history.jsonl  charts/`

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
rtk git add -A models/ && rtk git commit -m "chore: clean up legacy model files; retain only GW31 date-named per-position models"
```

---

## Task 10: Update docs and publish initial release

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/improvements-roadmap.md`

- [ ] **Step 1: Update `CLAUDE.md` Model Promotion section**

Replace the existing "Model Promotion via GitHub Releases" subsection with:

```markdown
### Model Promotion via GitHub Releases (Track I — Automated)

Promotion is now **fully automated** by `src/pipeline/promote.py`. Running `retrain` triggers the full pipeline:

1. Walk-forward evaluation on current season (per-GW ρ, MAE, hauler MAE)
2. Per-position benchmark comparison against `models/benchmark.json`
3. Positions that improve test ρ are promoted; others retain the current best model
4. Charts generated to `models/charts/` and attached to GitHub Release
5. `active_models.json` manifest published as a release asset
6. CI reads the manifest to download only the promoted model files

**Release tag format:** `model-YYYYMMDD` (e.g. `model-20260412`)

**Manual promotion (if needed):**
```bash
python -m src.pipeline.run retrain --gw <N>
```
No further steps needed — the pipeline handles benchmarking, release, and manifest automatically.

**To inspect the benchmark:**
```bash
cat models/benchmark.json
```

**To view metrics history:**
```bash
python -c "
import json
from pathlib import Path
for line in Path('models/metrics_history.jsonl').read_text().splitlines():
    r = json.loads(line)
    print(f\"{r['date']} {r['position']}: test_rho={r['test_rho']:.3f} promoted={r['promoted']}\")
"
```
```

- [ ] **Step 2: Update roadmap Track I status**

In `docs/improvements-roadmap.md`, change Track I status from `BACKLOG` to `COMPLETE (2026-04-12)` and add test count.

- [ ] **Step 3: Publish initial `model-20260412` release with current models**

This creates the first release under the new naming convention:

```bash
cd D:/FPL/fpl-assistant
# Generate charts from existing benchmark (single data point)
python -c "
import json
from pathlib import Path
from src.pipeline.promote import generate_charts, build_active_models_manifest
import joblib

models = {pos: joblib.load(f'models/rf_{pos.lower()}_20260412.sav') for pos in ['GK','DEF','MID','FWD']}
from src.pipeline.predict import ALL_FEATURE_COLUMNS

# Single-point history from benchmark
bench = json.loads(Path('models/benchmark.json').read_text())
history = [
    {'date': bench[pos]['date'], 'position': pos,
     'test_rho': bench[pos]['test_rho'], 'test_mae': bench[pos]['test_mae'],
     'per_gw_metrics': [], 'per_gw': []}
    for pos in bench
]
charts = generate_charts(history=history, models=models, feature_cols=ALL_FEATURE_COLUMNS, output_dir=Path('models/charts'))
print('Charts:', charts)
"
```

Then publish:

```bash
python -c "
import json
from pathlib import Path
from src.pipeline.promote import build_active_models_manifest, publish_release

bench = json.loads(Path('models/benchmark.json').read_text())
promoted = {pos: {**info, 'promoted': True} for pos, info in bench.items()}
manifest = build_active_models_manifest(promoted, published_date='2026-04-12')

import glob
charts = glob.glob('models/charts/*.png')
model_files = [f'models/rf_{pos.lower()}_20260412.sav' for pos in ['GK','DEF','MID','FWD']]

url = publish_release(
    date_str='20260412',
    model_files=model_files,
    manifest=manifest,
    chart_files=charts,
    metrics_summary={pos: {**info, 'promoted': True, 'algorithm': 'rf'} for pos, info in bench.items()},
    models_dir=Path('models'),
    dry_run=False,
)
print('Release:', url)
"
```

- [ ] **Step 4: Run full test suite one final time**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: 260+ passed.

- [ ] **Step 5: Final commit and push**

```bash
rtk git add CLAUDE.md docs/improvements-roadmap.md && rtk git commit -m "docs: update model promotion docs and mark Track I complete"
rtk git push
```

---

## Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Current season has no completed GWs (early season) | `test_gw_elapsed < MIN_TEST_GWS (6)` → `compare_to_benchmark` skips all promotion. Retrain still runs; metrics are logged. |
| `gh` CLI not authenticated in local env | `publish_release(dry_run=True)` skips `gh` call; use for testing. |
| Matplotlib not in requirements.txt | Task 5 Step 1 checks and adds it if missing. |
| Legacy `gw*` CI release tag | CI step falls back to `*.sav` glob download if no manifest found — backward compatible. |
| `get_active_models()` called at runtime | Lazy function (not import-time). If `active_models.json` is absent/malformed, falls back to hardcoded `ACTIVE_MODELS`. |
| Models dir git-ignored | Task 2 Step 3 explicitly adds `!models/benchmark.json` and `!models/metrics_history.jsonl` to `.gitignore`. |
| Same-date re-run | `publish_release` detects and deletes the existing tag before re-creating. |
| Non-promoted position's `.sav` missing from models/ | Hard `FileNotFoundError` raised before `publish_release` — prevents publishing an incomplete manifest. |
| RF train metrics biased (in-sample) | `oob_score=True` added to RF in Task 7; `evaluate_current_season` uses `oob_prediction_` when available; falls back to in-sample with a warning. |

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

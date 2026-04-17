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
_DEFAULT_HAULER_THRESHOLD = 5  # fallback when position is not passed


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
        if hasattr(model, "oob_prediction_") and model.oob_prediction_ is not None and len(model.oob_prediction_) == len(train_df):
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
        if len(gw_df) < 1:
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
        ax.barh(range(len(imp_vals)), imp_vals, color=colors.get(pos, "#999"))
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

    tmp_dir = None
    try:
        # Write manifest to a temp directory using the canonical filename so that
        # `gh release create` uploads it as "active_models.json" (not a random
        # NamedTemporaryFile prefix like "tmpXXXXXX_active_models.json").
        tmp_dir = tempfile.mkdtemp()
        manifest_path = Path(tmp_dir) / "active_models.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        assets = [str(manifest_path)] + model_files + chart_files
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
        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


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

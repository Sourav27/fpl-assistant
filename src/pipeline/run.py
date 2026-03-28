# src/pipeline/run.py
"""CLI entry point for the FPL weekly pipeline.

Usage:
    python -m src.pipeline.run pre-deadline   # Phase 1: fetch data + capture xP
    python -m src.pipeline.run predict        # Phase 2: generate predictions + optimize
    python -m src.pipeline.run post-gw        # Phase 3: collect results + live data patch
    python -m src.pipeline.run retrain        # Phase 4: retrain model (manual)
    python -m src.pipeline.run full           # Run phases 1-2 (for pre-deadline workflow)
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    VAASTAV_DIR, RESULTS_DIR, MODELS_DIR, CURRENT_SEASON,
    ACTIVE_MODEL, BOOTSTRAP_MAX_AGE_HOURS,
)
from src.pipeline.fetch import (
    fetch_bootstrap, get_current_gw, get_next_deadline,
    extract_xp_snapshot, fetch_fixtures, fetch_live_gw_data,
)
from src.pipeline.prepare import build_merged_dataset
from src.pipeline.features import engineer_features
from src.pipeline.predict import predict_next_gw, get_feature_columns
from src.pipeline.availability import filter_availability
from src.pipeline.optimize import optimize_team

logger = logging.getLogger(__name__)


def _load_cached_bootstrap(target_gw: int | None = None) -> dict | None:
    """Try to load a recent cached bootstrap snapshot."""
    snapshot_dir = RESULTS_DIR / "snapshots"
    if not snapshot_dir.exists():
        return None

    if target_gw:
        path = snapshot_dir / f"bootstrap_gw{target_gw}.json"
        if path.exists():
            return json.loads(path.read_text())

    # Find most recent snapshot
    snapshots = sorted(snapshot_dir.glob("bootstrap_gw*.json"), reverse=True)
    if not snapshots:
        return None

    path = snapshots[0]
    age_hours = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600
    if age_hours > BOOTSTRAP_MAX_AGE_HOURS:
        logger.warning(f"Cached bootstrap is {age_hours:.0f}h old (>{BOOTSTRAP_MAX_AGE_HOURS}h), skipping")
        return None

    return json.loads(path.read_text())


def phase_pre_deadline():
    """Phase 1: Fetch bootstrap data and capture xP before deadline."""
    print("[pre-deadline] Fetching FPL API bootstrap...")
    try:
        bootstrap = fetch_bootstrap()
    except Exception as e:
        logger.error(f"API fetch failed: {e}")
        bootstrap = _load_cached_bootstrap()
        if bootstrap is None:
            print("[pre-deadline] ERROR: API unreachable and no valid cached bootstrap. Aborting.")
            return None
        print("[pre-deadline] Using cached bootstrap snapshot")

    gw = get_current_gw(bootstrap)
    next_gw, deadline = get_next_deadline(bootstrap)
    print(f"[pre-deadline] Current GW: {gw}, Next deadline: GW{next_gw} at {deadline}")

    # Capture xP snapshot
    xp = extract_xp_snapshot(bootstrap)
    xp_path = VAASTAV_DIR / "data" / CURRENT_SEASON / "gws" / f"xP{next_gw}.csv"
    xp_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(xp.items()), columns=["id", "xP"]).to_csv(xp_path, index=False)
    print(f"[pre-deadline] Saved xP snapshot for GW{next_gw} ({len(xp)} players)")

    # Save bootstrap for reference
    snapshot_dir = RESULTS_DIR / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with open(snapshot_dir / f"bootstrap_gw{next_gw}.json", "w") as f:
        json.dump(bootstrap, f)
    print("[pre-deadline] Saved bootstrap snapshot")

    return next_gw


def phase_predict(target_gw: int | None = None):
    """Phase 2: Build features, predict, filter availability, optimize."""
    print("[predict] Building merged dataset...")
    merged = build_merged_dataset(vaastav_dir=VAASTAV_DIR)
    print(f"[predict] Dataset: {len(merged)} rows, {len(merged.columns)} columns")

    print("[predict] Engineering features...")
    features = engineer_features(merged)
    print(f"[predict] Features: {len(features)} rows after NaN drop")

    # Get latest row per player for prediction
    latest = features.sort_values(["element", "GW"]).groupby("element").last().reset_index()

    # Ensure now_cost column exists (vaastav uses 'value', FPL API uses 'now_cost')
    if "now_cost" not in latest.columns:
        latest["now_cost"] = latest.get("value", pd.Series(50, index=latest.index))

    # Load bootstrap for cost override and availability filtering
    bootstrap = None
    if target_gw:
        bootstrap = _load_cached_bootstrap(target_gw)
        if bootstrap:
            cost_map = {e["id"]: e["now_cost"] for e in bootstrap["elements"]}
            latest["now_cost"] = latest["element"].map(cost_map).fillna(latest["now_cost"])

    print("[predict] Generating predictions...")
    model_path = ACTIVE_MODEL
    if not model_path.exists():
        print(f"[predict] WARNING: Model not found at {model_path}. Using xP from API.")
        if target_gw:
            xp_path = VAASTAV_DIR / "data" / CURRENT_SEASON / "gws" / f"xP{target_gw}.csv"
            if xp_path.exists():
                xp_df = pd.read_csv(xp_path)
                latest = latest.merge(
                    xp_df.rename(columns={"id": "element"}),
                    on="element", how="left", suffixes=("_feat", ""),
                )
        predictions = latest[["element", "name", "position", "team"]].copy()
        predictions["xP"] = latest["xP"] if "xP" in latest.columns else 0
        predictions["now_cost"] = latest["now_cost"]
    else:
        predictions = predict_next_gw(latest, model_path)

    # Apply availability filtering
    if bootstrap:
        print("[predict] Filtering by player availability...")
        before_count = len(predictions)
        predictions = filter_availability(predictions, bootstrap)
        excluded = before_count - len(predictions)
        if excluded > 0:
            print(f"[predict] Excluded {excluded} unavailable players")
    else:
        print("[predict] Skipping availability filter (no bootstrap data)")

    print("[predict] Optimizing team selection...")

    # Save results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gw_label = f"gw{target_gw}" if target_gw else "latest"

    try:
        result = optimize_team(predictions)
    except Exception as e:
        logger.warning(f"optimize_team failed (likely infeasible LP — small player pool): {e}")
        empty = pd.DataFrame(columns=["element", "name", "position", "team", "xP", "now_cost"])
        empty.to_csv(RESULTS_DIR / f"xi_{gw_label}.csv", index=False)
        empty.to_csv(RESULTS_DIR / f"squad_{gw_label}.csv", index=False)
        print(f"[predict] Optimization infeasible — saved empty CSVs for {gw_label}")
        return {"xi": empty, "squad": empty, "captain": None, "vice_captain": None, "total_xp": 0.0}

    result["xi"].to_csv(RESULTS_DIR / f"xi_{gw_label}.csv", index=False)
    result["squad"].to_csv(RESULTS_DIR / f"squad_{gw_label}.csv", index=False)

    print(f"\n{'='*50}")
    print(f"OPTIMAL XI for GW{target_gw or '?'}:")
    print(f"{'='*50}")
    xi = result["xi"].sort_values("position")
    for _, p in xi.iterrows():
        cap = " (C)" if p["element"] == result["captain"]["element"] else ""
        vc = " (VC)" if p["element"] == result["vice_captain"]["element"] else ""
        print(f"  {p['position']:3s} | {p['name']:20s} | {p['team']:15s} | xP: {p['xP']:.1f}{cap}{vc}")
    print(f"\nTotal xP (with captain): {result['total_xp']:.1f}")
    print(f"Budget used: {result['squad']['now_cost'].sum() / 10:.1f}M")

    return result


def phase_post_gw():
    """Phase 3: Collect actual results and save live GW data."""
    print("[post-gw] Fetching updated bootstrap...")
    try:
        bootstrap = fetch_bootstrap()
    except Exception as e:
        logger.error(f"API fetch failed during post-gw: {e}")
        print("[post-gw] ERROR: API unreachable. Skipping live data collection.")
        return

    gw = get_current_gw(bootstrap)
    print(f"[post-gw] Current GW: {gw}")

    # Fetch fixtures for actual scores
    print("[post-gw] Fetching fixtures...")
    fixtures = fetch_fixtures()
    finished = [f for f in fixtures if f.get("finished") and f.get("event") == gw]
    print(f"[post-gw] {len(finished)} finished fixtures in GW{gw}")

    # Collect live player data for this GW
    print(f"[post-gw] Fetching player histories for GW{gw}...")
    live_df = fetch_live_gw_data(target_gw=gw, bootstrap_data=bootstrap)

    if not live_df.empty:
        gw_dir = VAASTAV_DIR / "data" / CURRENT_SEASON / "gws"
        gw_dir.mkdir(parents=True, exist_ok=True)
        live_path = gw_dir / f"gw{gw}_live.csv"
        live_df.to_csv(live_path, index=False)
        print(f"[post-gw] Saved {len(live_df)} player rows to {live_path}")
    else:
        print("[post-gw] No player data collected (GW may not be finished)")

    print("[post-gw] Done. Run 'predict' to update features with new data.")


def phase_retrain(target_gw: int | None = None):
    """Phase 4: Retrain RF model on full dataset (manual trigger)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    import joblib

    print("[retrain] Building full feature-engineered dataset...")
    merged = build_merged_dataset(vaastav_dir=VAASTAV_DIR)
    features = engineer_features(merged)
    print(f"[retrain] Training data: {len(features)} rows")

    feature_cols = get_feature_columns()
    X = features[feature_cols].fillna(0)
    y = features["total_points"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[retrain] Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"[retrain] New model — MAE: {mae:.2f}, R2: {r2:.3f}")

    # Compare with existing model if available
    if ACTIVE_MODEL.exists():
        old_model = joblib.load(ACTIVE_MODEL)
        old_pred = old_model.predict(X_test)
        old_mae = mean_absolute_error(y_test, old_pred)
        old_r2 = r2_score(y_test, old_pred)
        print(f"[retrain] Old model — MAE: {old_mae:.2f}, R2: {old_r2:.3f}")
        if mae < old_mae:
            print("[retrain] New model is BETTER (lower MAE)")
        else:
            print("[retrain] New model is WORSE (higher MAE) — consider keeping old model")

    # Save with GW label (or timestamp fallback)
    label = f"gw{target_gw}" if target_gw else datetime.now().strftime("%Y%m%d_%H%M%S")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    new_path = MODELS_DIR / f"rf_model_{label}.sav"
    joblib.dump(model, new_path)
    print(f"[retrain] Saved new model to {new_path}")
    print(f"[retrain] To promote: update ACTIVE_MODEL in src/config.py to point to {new_path.name}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="FPL Weekly Pipeline")
    parser.add_argument("phase", choices=["pre-deadline", "predict", "post-gw", "retrain", "full"],
                        help="Pipeline phase to run")
    parser.add_argument("--gw", type=int, help="Target gameweek (optional)")
    args = parser.parse_args()

    if args.phase == "pre-deadline":
        phase_pre_deadline()
    elif args.phase == "predict":
        phase_predict(args.gw)
    elif args.phase == "post-gw":
        phase_post_gw()
    elif args.phase == "retrain":
        phase_retrain(args.gw)
    elif args.phase == "full":
        gw = phase_pre_deadline()
        if gw:
            phase_predict(gw)


if __name__ == "__main__":
    main()

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
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from src.config import (
    VAASTAV_DIR, RESULTS_DIR, MODELS_DIR, CURRENT_SEASON,
    ACTIVE_MODEL, ACTIVE_MODELS, BOOTSTRAP_MAX_AGE_HOURS, SNAPSHOTS_DIR,
    FPL_ENTRY_URL, load_user_config, UserConfigError,
)
from src.pipeline.fetch import (
    fetch_bootstrap, get_current_gw, get_next_deadline,
    extract_xp_snapshot, fetch_fixtures, fetch_live_gw_data,
    find_wayback_snapshot, fetch_wayback_bootstrap,
    _api_get_with_retry, ELEMENT_TYPE_MAP,
)
from src.pipeline.user import fetch_user_team_state
from src.pipeline.recommend import recommend_transfers, recommend_wildcard, save_recommend_csv
from src.pipeline.prepare import build_merged_dataset
from src.pipeline.analysis import (
    compute_prediction_misses, compute_dream_team,
    format_post_match_summary, append_accuracy_log,
)
from src.pipeline.features import engineer_features, build_prediction_features
from src.pipeline.predict import (
    predict_next_gw, get_feature_columns, save_full_predictions, apply_xp_corrections,
    load_position_models, predict_next_gw_per_position, ALL_FEATURE_COLUMNS,
)
from src.pipeline.availability import filter_availability
from src.pipeline.optimize import optimize_team

logger = logging.getLogger(__name__)

# Daily CI refresh runs at 07:06 IST (01:36 UTC). A bootstrap snapshot is
# considered fresh if it was saved after the most recent 07:06 IST cutoff.
_IST = timezone(timedelta(hours=5, minutes=30))


def _latest_bootstrap_cutoff() -> datetime:
    """Return the most recent 07:06 IST as a UTC-aware datetime."""
    now_ist = datetime.now(_IST)
    cutoff = now_ist.replace(hour=7, minute=6, second=0, microsecond=0)
    if now_ist < cutoff:
        cutoff -= timedelta(days=1)
    return cutoff.astimezone(timezone.utc)


def _score_from_entry_picks(entry_picks: dict) -> int:
    """Extract the user's actual GW score from FPL entry picks response.

    Uses entry_history.points which already accounts for captain multiplier,
    auto-subs, VC activation, and bench boost — unlike reconstructing from
    per-player actual_points sums which miss all of these.
    """
    return entry_picks["entry_history"]["points"]


def _filter_gw_transfers(rec_df: pd.DataFrame, current_gw: int) -> pd.DataFrame:
    """Filter recommendation df to current-GW transfers only.

    Prevents future-horizon transfers (e.g. GW32 Walker when analysing GW31)
    from leaking into post-match recommended squad comparisons.
    """
    return rec_df[rec_df["gw"] == current_gw]


def _load_cached_bootstrap(target_gw: int | None = None) -> dict | None:
    """Load a cached bootstrap snapshot if it is fresh enough.

    Freshness is determined by the daily CI cutoff: a snapshot must have been
    saved after the most recent 07:06 IST (the scheduled daily refresh time).
    This is stricter than a fixed 48-hour window — a snapshot from yesterday
    afternoon is stale by morning even if < 24 h old.
    """
    snapshot_dir = SNAPSHOTS_DIR
    if not snapshot_dir.exists():
        return None

    cutoff = _latest_bootstrap_cutoff()

    if target_gw:
        path = snapshot_dir / f"bootstrap_gw{target_gw}.json"
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if mtime >= cutoff:
                return json.loads(path.read_text(encoding="utf-8"))
            logger.warning(
                f"bootstrap_gw{target_gw}.json is stale "
                f"(saved {mtime:%Y-%m-%d %H:%M} UTC, cutoff {cutoff:%Y-%m-%d %H:%M} UTC)"
            )
            return None

    # No target_gw — use the most recent snapshot if fresh
    snapshots = sorted(snapshot_dir.glob("bootstrap_gw*.json"), reverse=True)
    if not snapshots:
        return None

    path = snapshots[0]
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    if mtime < cutoff:
        logger.warning(
            f"Most recent bootstrap snapshot is stale "
            f"(saved {mtime:%Y-%m-%d %H:%M} UTC, cutoff {cutoff:%Y-%m-%d %H:%M} UTC)"
        )
        return None

    return json.loads(path.read_text(encoding="utf-8"))


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
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOTS_DIR / f"bootstrap_gw{next_gw}.json", "w") as f:
        json.dump(bootstrap, f)
    print("[pre-deadline] Saved bootstrap snapshot")

    return next_gw


def phase_predict(target_gw: int | None = None):
    """Phase 2: Build features, predict, filter availability, optimize."""
    print("[predict] Building merged dataset...")
    merged = build_merged_dataset(vaastav_dir=VAASTAV_DIR)
    print(f"[predict] Dataset: {len(merged)} rows, {len(merged.columns)} columns")

    if not merged.empty:
        print("[predict] Engineering features...")
        features = engineer_features(merged)
        print(f"[predict] Features: {len(features)} rows after NaN drop")
        # Group by persistent code when available to avoid element-ID recycling across seasons.
        player_id = "code" if "code" in features.columns else "element"
        latest = features.sort_values([player_id, "GW"]).groupby(player_id).last().reset_index()
        # Ensure now_cost column exists (vaastav uses 'value', FPL API uses 'now_cost')
        if "now_cost" not in latest.columns:
            latest["now_cost"] = latest.get("value", pd.Series(50, index=latest.index))
    else:
        # No vaastav data — bootstrap ep_next is used as the xP source (seeded below).
        print("[predict] No historical data — will seed xP from bootstrap ep_next")
        latest = pd.DataFrame()
        player_id = "code"

    # Training cutoff callout — warn if predicting for a GW the model was trained on.
    _model_gw_match = re.search(r"gw(\d+)", ACTIVE_MODEL.stem, re.IGNORECASE)
    _training_gw = int(_model_gw_match.group(1)) if _model_gw_match else None
    if _training_gw and target_gw and target_gw <= _training_gw:
        print(
            f"[predict] NOTE: Model rf_model_gw{_training_gw}.sav was trained through GW{_training_gw}. "
            f"Predictions for GW{target_gw} use in-sample data — treat results as validation, "
            f"not genuine out-of-sample forecasts."
        )

    # Load bootstrap for metadata + availability filtering.
    bootstrap = _load_cached_bootstrap(target_gw)
    if bootstrap is None:
        print("[predict] No valid cached bootstrap — fetching from live API...")
        try:
            live_bootstrap = fetch_bootstrap()
            live_gw = get_current_gw(live_bootstrap)
            if target_gw and live_gw and target_gw < live_gw:
                # Historical GW: live API has wrong squad/cost data. Try Wayback Machine.
                deadline = next(
                    (e["deadline_time"] for e in live_bootstrap["events"] if e["id"] == target_gw),
                    None,
                )
                if deadline:
                    print(f"[predict] GW{target_gw} is historical — searching Wayback Machine for pre-deadline snapshot...")
                    ts = find_wayback_snapshot(deadline)
                    if ts:
                        print(f"[predict] Found Wayback snapshot {ts} — downloading...")
                        try:
                            bootstrap = fetch_wayback_bootstrap(ts)
                            SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
                            with open(SNAPSHOTS_DIR / f"bootstrap_gw{target_gw}.json", "w", encoding="utf-8") as f:
                                json.dump(bootstrap, f)
                            print(f"[predict] Cached Wayback bootstrap as bootstrap_gw{target_gw}.json")
                        except Exception as e:
                            logger.warning(f"Wayback bootstrap download failed: {e}")
                    else:
                        print(
                            f"[predict] WARNING: No Wayback snapshot found for GW{target_gw} "
                            f"(deadline {deadline}). Proceeding without bootstrap — "
                            f"player metadata and blank-GW corrections will be missing."
                        )
                else:
                    print(f"[predict] WARNING: GW{target_gw} not found in live bootstrap events.")
            else:
                bootstrap = live_bootstrap
                if target_gw:
                    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
                    with open(SNAPSHOTS_DIR / f"bootstrap_gw{target_gw}.json", "w", encoding="utf-8") as f:
                        json.dump(bootstrap, f)
                    print(f"[predict] Cached live bootstrap as bootstrap_gw{target_gw}.json")
        except Exception as e:
            logger.warning(f"Could not fetch bootstrap from API: {e}. Proceeding without (stale data risk).")

    if bootstrap and player_id == "code":
        # Override stale historical metadata (name/position/team/element/cost)
        # with current-season values from the FPL API bootstrap.
        team_map = {t["id"]: t["name"] for t in bootstrap.get("teams", [])}
        bs_df = pd.DataFrame([{
            "code": e["code"],
            "element": e["id"],
            "name": e["web_name"],
            "position": ELEMENT_TYPE_MAP.get(e["element_type"], "MID"),
            "team": team_map.get(e["team"], ""),
            "now_cost": e["now_cost"],
        } for e in bootstrap["elements"]])
        # Drop stale metadata; re-join from bootstrap keyed on persistent code.
        if not latest.empty:
            stale = [c for c in ["element", "name", "position", "team", "now_cost"]
                     if c in latest.columns]
            latest = latest.drop(columns=stale).merge(bs_df, on="code", how="left")
        # No historical data (vaastav not cloned): seed directly from bootstrap with ep_next xP.
        if latest.empty:
            print("[predict] No historical rows — seeding from bootstrap ep_next")
            ep_map = {e["code"]: float(e.get("ep_next") or 0) for e in bootstrap["elements"]}
            latest = bs_df.copy()
            latest["xP"] = latest["code"].map(ep_map).fillna(0)
        # Drop players not in current bootstrap (retired / transferred abroad).
        in_bootstrap = latest["element"].notna()
        n_excluded = (~in_bootstrap).sum()
        if n_excluded > 0:
            print(f"[predict] Excluding {n_excluded} historical players not in current FPL season")
        latest = latest[in_bootstrap].copy()
        latest["now_cost"] = latest["now_cost"].fillna(50)
        latest["position"] = latest["position"].fillna("MID")
        latest["name"] = latest["name"].fillna("Unknown")
        latest["team"] = latest["team"].fillna("Unknown")
        latest["element"] = latest["element"].astype(int)
    elif bootstrap:
        # Fallback (no code column): legacy cost-only override by element.
        cost_map = {e["id"]: e["now_cost"] for e in bootstrap["elements"]}
        latest["now_cost"] = latest["element"].map(cost_map).fillna(latest["now_cost"])

    print("[predict] Generating predictions...")

    # Attempt per-position prediction (Track B)
    pos_models = load_position_models()
    any_model_available = any(m is not None for m in pos_models.values())

    if any_model_available and not latest.empty:
        print("[predict] Using per-position models (Track B)")
        ep_next_map = {}
        if bootstrap:
            ep_next_map = {
                el["id"]: float(el.get("ep_next") or 0)
                for el in bootstrap.get("elements", [])
            }
        # Expand to one row per fixture (DGW players get 2 rows).
        player_features = latest.copy()
        if bootstrap and target_gw:
            try:
                fixtures = fetch_fixtures()
                teams_list = bootstrap.get("teams", [])
                expanded = build_prediction_features(latest, fixtures, target_gw, teams_list)
                if not expanded.empty:
                    dgw_count = (expanded.groupby("element").size() > 1).sum()
                    if dgw_count:
                        print(f"[predict] Expanded {dgw_count} DGW players to 2 rows each")
                    player_features = expanded
            except Exception as exc:
                logger.warning(f"[predict] DGW expansion failed ({exc}) — using single-row predictions")

        predictions = predict_next_gw_per_position(
            player_features,
            models=pos_models,
            ep_next_map=ep_next_map,
        )
        # Guard: per-position models may only cover a subset of positions (e.g. only
        # "MID" in the manifest) or may fail silently due to sklearn version mismatch.
        # If any positions from latest are missing from predictions, fill them in from
        # latest using ep_next xP so the optimizer always receives a full player pool.
        if not latest.empty:
            covered = set(predictions["element"].tolist()) if not predictions.empty else set()
            missing_rows = latest[~latest["element"].isin(covered)].copy()
            if not missing_rows.empty:
                print(
                    f"[predict] WARNING: {len(missing_rows)} players missing from per-position "
                    f"predictions (positions not in manifest or prediction failed) — "
                    f"filling from ep_next"
                )
                missing_rows["xP"] = missing_rows["xP"] if "xP" in missing_rows.columns else 0.0
                missing_rows["_fallback"] = True
                fill_cols = [c for c in predictions.columns if c in missing_rows.columns]
                if predictions.empty:
                    predictions = missing_rows[fill_cols].reset_index(drop=True)
                else:
                    predictions = pd.concat(
                        [predictions, missing_rows[fill_cols]], ignore_index=True
                    )
    else:
        if any_model_available:
            print("[predict] No historical feature rows — falling back to global model or ep_next")
        else:
            print("[predict] No per-position models found — falling back to global model")

        model_path = ACTIVE_MODEL
        _fallback = False
        if not model_path.exists() or latest.empty:
            print(f"[predict] WARNING: Model not found or no data. Using xP from API.")
            _fallback = True
        else:
            try:
                predictions = predict_next_gw(latest, model_path)
            except ValueError as e:
                print(
                    f"[predict] WARNING: Stale model at {model_path} is incompatible "
                    f"({e}). Run `retrain` to rebuild. Falling back to API xP."
                )
                _fallback = True

        if _fallback:
            if target_gw:
                xp_path = VAASTAV_DIR / "data" / CURRENT_SEASON / "gws" / f"xP{target_gw}.csv"
                if xp_path.exists():
                    xp_df = pd.read_csv(xp_path)
                    latest = latest.merge(
                        xp_df.rename(columns={"id": "element"}),
                        on="element", how="left", suffixes=("_feat", ""),
                    )
            predictions = latest[["element", "name", "position", "team"]].copy()
            predictions["xP"] = (latest["xP"] if "xP" in latest.columns else 0)
            predictions["xP"] = predictions["xP"].fillna(0).clip(lower=0)
            predictions["now_cost"] = latest["now_cost"].fillna(50)

    # Apply xP corrections: blank GW zeroing (A-F4)
    if bootstrap and target_gw:
        blank_count_before = (predictions["xP"] > 0).sum()
        predictions = apply_xp_corrections(predictions, bootstrap, target_gw)
        blank_count_after = (predictions["xP"] > 0).sum()
        blanked = blank_count_before - blank_count_after
        if blanked > 0:
            print(f"[predict] Zeroed xP for {blanked} blank-GW players")

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

    # Save full predictions for recommend + analysis phases
    pred_path = RESULTS_DIR / f"predictions_{gw_label}.csv"
    save_full_predictions(predictions, pred_path)
    print(f"[predict] Saved full predictions ({len(predictions)} players) to {pred_path}")

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

    # P2a: post-match analysis (skipped gracefully if user config missing)
    try:
        cfg = load_user_config()
    except UserConfigError:
        print("[post-gw] user_config.yaml not found — skipping post-match analysis")
        return

    entry_id = cfg["teams"]["default"]["entry_id"]

    # Load predictions for this GW
    gw_label = f"gw{gw}"
    pred_path = RESULTS_DIR / f"predictions_{gw_label}.csv"
    if not pred_path.exists():
        print(f"[post-gw] Predictions file {pred_path} not found — skipping analysis")
        return

    predictions = pd.read_csv(pred_path)

    # Fetch user picks for this GW — A-F1: your_pts from entry_history.points (correct)
    your_pts = 0
    your_xp = 0.0
    misses = []
    your_picks = pd.DataFrame()

    try:
        entry_picks_data = _api_get_with_retry(
            f"{FPL_ENTRY_URL}/{entry_id}/event/{gw}/picks/"
        ).json()
        your_pts = _score_from_entry_picks(entry_picks_data)  # A-F1: captain + auto-subs correct
        picks_elements = {p["element"] for p in entry_picks_data.get("picks", [])}
        your_picks = predictions[predictions["element"].isin(picks_elements)].copy()
    except Exception as e:
        logger.warning(f"Could not fetch user picks: {e}")

    if not live_df.empty and not your_picks.empty:
        actual_map = live_df.set_index("element")["total_points"].to_dict()
        your_picks["actual_points"] = your_picks["element"].map(actual_map).fillna(0)
        your_xp = float(your_picks["xP"].sum())
        misses = compute_prediction_misses(your_picks)

    # Recommended team comparison — use saved squad CSV for accuracy
    recommended_pts = None
    recommended_xp = None
    squad_rec_path = RESULTS_DIR / f"squad_recommend_{gw_label}.csv"
    if squad_rec_path.exists() and not live_df.empty:
        rec_squad_df = pd.read_csv(squad_rec_path)
        actual_map_rec = live_df.set_index("element")["total_points"].to_dict()
        rec_squad_df["actual_points"] = rec_squad_df["element"].map(actual_map_rec).fillna(0)
        recommended_pts = int(rec_squad_df["actual_points"].sum())
        recommended_xp = float(rec_squad_df["xP"].sum()) if "xP" in rec_squad_df.columns else None

    # Dream team from live data
    dream_pts = None
    if not live_df.empty:
        try:
            dream = compute_dream_team(live_df)
            dream_pts = int(dream["total_points"].sum() if "total_points" in dream.columns
                            else dream["xP"].sum())
        except Exception as e:
            logger.warning(f"Dream team computation failed: {e}")

    # Benchmarks
    from src.pipeline.user import fetch_gw_benchmarks
    overall_league_id = None
    try:
        entry_data = _api_get_with_retry(f"{FPL_ENTRY_URL}/{entry_id}/").json()
        for league in entry_data.get("leagues", {}).get("classic", []):
            if league.get("league_type") == "s" and league.get("scoring") == "c":
                overall_league_id = league["id"]
                break
    except Exception:
        pass

    benchmarks = {}
    your_percentile_rank = None
    if overall_league_id:
        try:
            benchmarks = fetch_gw_benchmarks(gw, bootstrap, overall_league_id)
        except Exception as e:
            logger.warning(f"Could not fetch benchmarks: {e}")
    # Percentile rank from history
    try:
        history = _api_get_with_retry(f"{FPL_ENTRY_URL}/{entry_id}/history/").json()
        for row in history.get("current", []):
            if row["event"] == gw:
                your_percentile_rank = row.get("percentile_rank")
                break
    except Exception:
        pass

    # Print summary
    print(format_post_match_summary(
        gw=gw, your_pts=your_pts, your_xp=your_xp,
        recommended_pts=recommended_pts, recommended_xp=recommended_xp,
        dream_pts=dream_pts, benchmarks=benchmarks,
        your_percentile_rank=your_percentile_rank, misses=misses,
    ))

    # Wildcard baseline: full 15-player optimizer squad summed against live data.
    # Intentionally the full squad (not just XI) so it reflects bench-boost potential.
    wildcard_pts = None
    wildcard_xp = None
    squad_path = RESULTS_DIR / f"squad_{gw_label}.csv"
    if squad_path.exists() and not live_df.empty:
        squad_df = pd.read_csv(squad_path)
        actual_map_wc = live_df.set_index("element")["total_points"].to_dict()
        squad_df["actual_points"] = squad_df["element"].map(actual_map_wc).fillna(0)
        wildcard_pts = int(squad_df["actual_points"].sum())
        wildcard_xp = float(squad_df["xP"].sum()) if "xP" in squad_df.columns else None

    # Write accuracy log
    log_path = RESULTS_DIR / "accuracy_log.csv"
    append_accuracy_log(
        path=log_path, gw=gw,
        your_pts=your_pts, your_xp=your_xp,
        recommended_pts=recommended_pts, recommended_xp=recommended_xp,
        wildcard_pts=wildcard_pts, wildcard_xp=wildcard_xp,
        dream_pts=dream_pts, your_percentile_rank=your_percentile_rank,
        benchmarks=benchmarks, ranked_count=benchmarks.get("ranked_count"),
        picks_df=your_picks if not your_picks.empty else None,
    )
    print(f"[post-gw] Accuracy log updated: {log_path}")


def _is_wildcard_mode(user_state, wildcard_flag: bool) -> bool:
    """Return True only when --wildcard flag is explicitly passed. Active chip auto-detection
    is deferred — recommend_transfers always runs the normal ILP unless explicitly overridden."""
    return wildcard_flag


def phase_recommend(
    target_gw: int | None = None,
    team_key: str = "default",
    horizon: int | None = None,
    wildcard: bool = False,
) -> dict | None:
    """Recommend phase: fetch user state, load predictions, run transfer optimizer."""
    # Load user config
    try:
        cfg = load_user_config()
    except UserConfigError as e:
        print(f"[recommend] ERROR: {e}")
        return None

    entry_id = cfg["teams"][team_key]["entry_id"]
    prefs = cfg["preferences"]
    horizon = horizon or prefs["horizon_gws"]
    fdr_sensitivity = prefs["fdr_sensitivity"]
    max_hit_points = prefs["max_hit_points"]

    # Load predictions
    gw_label = f"gw{target_gw}" if target_gw else "latest"
    pred_path = RESULTS_DIR / f"predictions_{gw_label}.csv"
    if not pred_path.exists():
        print(f"[recommend] ERROR: Predictions not found at {pred_path}. Run 'predict' first.")
        return None

    predictions = pd.read_csv(pred_path)
    print(f"[recommend] Loaded {len(predictions)} player predictions from {pred_path}")

    if predictions.empty:
        print(
            "[recommend] ERROR: predictions file is empty — optimizer cannot run. "
            "Re-run 'predict' to regenerate predictions."
        )
        return None

    # Fetch user team state
    print(f"[recommend] Fetching team state for entry {entry_id}...")
    try:
        bootstrap = _load_cached_bootstrap(target_gw)
        if bootstrap is None:
            bootstrap = fetch_bootstrap()
        # Picks only exist for completed GWs. target_gw is the upcoming GW we're
        # planning for, so fetch the squad from the most recently completed GW.
        current_gw = get_current_gw(bootstrap)
        squad_gw = current_gw or target_gw
        user_state = fetch_user_team_state(entry_id, squad_gw, bootstrap)
    except Exception as e:
        print(f"[recommend] ERROR fetching team state: {e}")
        return None

    print(f"[recommend] Team: {len(user_state.current_squad)} players, "
          f"bank £{user_state.bank/10:.1f}m, {user_state.free_transfers} FT(s)")

    # Fetch fixtures for FDR
    try:
        fixtures = fetch_fixtures()
    except Exception:
        fixtures = []
        print("[recommend] WARNING: Could not fetch fixtures. FDR weighting disabled.")

    # Run optimizer
    if _is_wildcard_mode(user_state, wildcard):
        chip_name = user_state.active_chip or "wildcard (flag)"
        print(f"[recommend] {chip_name} active — running unconstrained squad selection")
        plan = recommend_wildcard(user_state, predictions)
    else:
        plan = recommend_transfers(
            user_state=user_state,
            predictions=predictions,
            fixtures=fixtures,
            horizon=horizon,
            fdr_sensitivity=fdr_sensitivity,
            max_hit_points=max_hit_points,
        )

    # Print summary
    print(f"\n{'='*50}")
    print(f"TRANSFER RECOMMENDATIONS (GW{target_gw}, horizon={horizon})")
    print(f"{'='*50}")
    transfers_by_gw = plan.get("transfers", [])
    if isinstance(transfers_by_gw, list) and transfers_by_gw:
        for gw_offset, gw_data in enumerate(transfers_by_gw):
            gw = (target_gw or 0) + gw_offset
            t_list = gw_data if isinstance(gw_data, list) else gw_data.get("transfers", [])
            if t_list:
                for t in t_list:
                    print(f"  GW{gw}: OUT {t['player_out']} (£{t['price_out']:.1f}m) "
                          f"-> IN {t['player_in']} (£{t['price_in']:.1f}m)")
            else:
                print(f"  GW{gw}: Hold")
    print(f"\nProjected xP ({horizon} GWs): {plan.get('projected_xp', 0):.1f}")
    print(f"Transfer cost: {plan.get('hit_cost', 0)} points")
    print(f"Bank after: £{plan.get('bank_after', 0):.1f}m")

    # Save CSV
    out_path = RESULTS_DIR / f"recommend_{gw_label}.csv"
    save_recommend_csv(plan, out_path, start_gw=target_gw or 0)
    print(f"\nSaved to {out_path}")

    # Save post-transfer squad and XI for Discord notification
    # For wildcard/freehit: show the user's actual current squad (already shown as Wildcard XI).
    # For normal transfers: show squad_after (current squad + recommended moves applied).
    if _is_wildcard_mode(user_state, wildcard):
        squad_after_ids = user_state.current_squad
    else:
        squad_after_ids = plan.get("squad_after", [])
    if squad_after_ids:
        from src.pipeline.optimize import select_xi
        squad_rec = predictions[predictions["element"].isin(squad_after_ids)][
            ["element", "name", "position", "team", "now_cost", "xP"]
        ].reset_index(drop=True)
        # Free transfers remaining after recommended moves
        free_used = sum(
            1 for t in (plan.get("transfers") or [{}])[0].get("transfers", [])
            if t.get("hit_cost", 0) == 0
        )
        squad_rec["free_transfers_after"] = max(1, user_state.free_transfers - free_used)
        squad_rec_path = RESULTS_DIR / f"squad_recommend_{gw_label}.csv"
        squad_rec.to_csv(squad_rec_path, index=False)
        xi_rec = select_xi(squad_rec)
        xi_rec_path = RESULTS_DIR / f"xi_recommend_{gw_label}.csv"
        xi_rec.to_csv(xi_rec_path, index=False)
        print(f"Saved post-transfer squad to {squad_rec_path}")
        print(f"Saved post-transfer XI to {xi_rec_path}")

    return plan


def phase_retrain(target_gw: int | None = None):
    """Phase 4: Retrain 4 per-position RF models on full dataset (manual trigger)."""
    from datetime import date as _date
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from scipy.stats import spearmanr
    import joblib
    from src.pipeline.promote import run_promotion_pipeline
    from src.config import (BENCHMARK_PATH, METRICS_LEDGER_PATH,
                             CHARTS_DIR, CURRENT_SEASON)

    print("[retrain] Building full feature-engineered dataset...")
    merged = build_merged_dataset(vaastav_dir=VAASTAV_DIR)
    features = engineer_features(merged)
    print(f"[retrain] Training data: {len(features)} rows")

    # Normalize position to string label
    if "position" in features.columns and pd.api.types.is_integer_dtype(features["position"]):
        features["position"] = features["position"].map(ELEMENT_TYPE_MAP)

    feature_cols = [c for c in ALL_FEATURE_COLUMNS if c in features.columns]

    date_str = _date.today().strftime("%Y%m%d")
    algorithm = "rf"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    position_results = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_df = features[features["position"] == pos].copy()
        if len(pos_df) < 100:
            print(f"[retrain] {pos}: insufficient data ({len(pos_df)} rows) — skipping")
            continue

        X = pos_df[feature_cols].fillna(0)
        y = pos_df["total_points"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rho, _ = spearmanr(y_pred, y_test)

        new_path = MODELS_DIR / f"{algorithm}_{pos.lower()}_{date_str}.sav"
        joblib.dump(model, new_path)
        position_results[pos] = {"mae": mae, "rho": rho, "path": new_path, "n": len(pos_df), "model": model}
        print(f"[retrain] {pos}: MAE={mae:.3f}, Spearman rho={rho:.3f} ({len(pos_df)} rows) -> {new_path.name}")

    print("\n[retrain] Summary:")
    for pos, r in position_results.items():
        print(f"  {pos}: MAE={r['mae']:.3f}, rho={r['rho']:.3f}")

    if position_results:
        print("\n[retrain] Running promotion pipeline...")
        trained_models = {
            pos: (r["model"], r["path"])
            for pos, r in position_results.items()
            if "model" in r
        }
        run_promotion_pipeline(
            trained_models=trained_models,
            algorithm=algorithm,
            features_df=features,
            feature_cols=feature_cols,
            date_str=date_str,
            model_dir=MODELS_DIR,
            benchmark_path=BENCHMARK_PATH,
            ledger_path=METRICS_LEDGER_PATH,
            charts_dir=CHARTS_DIR,
            current_season=CURRENT_SEASON,
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="FPL Weekly Pipeline")
    parser.add_argument(
        "phase",
        choices=["pre-deadline", "predict", "post-gw", "retrain", "full", "recommend"],
        help="Pipeline phase to run",
    )
    parser.add_argument("--gw", type=int, help="Target gameweek (optional)")
    parser.add_argument("--horizon", type=int, help="GWs to plan ahead (1-5, default from config)")
    parser.add_argument("--wildcard", action="store_true", help="Ignore current squad (wildcard/FH mode)")
    parser.add_argument("--team", default="default", help="Which team from user_config.yaml (default/alt)")
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
    elif args.phase == "recommend":
        phase_recommend(
            target_gw=args.gw,
            team_key=args.team,
            horizon=args.horizon,
            wildcard=args.wildcard,
        )


if __name__ == "__main__":
    main()

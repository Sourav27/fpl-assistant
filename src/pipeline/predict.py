# src/pipeline/predict.py
"""Load trained models and generate predictions."""
import joblib
import logging
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from src.config import ACTIVE_MODEL, ACTIVE_MODELS, get_active_models
from src.pipeline.features import FIXTURE_FEATURE_COLUMNS

logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    "total_points_roll_4", "total_points_roll_8",
    "minutes_roll_4", "minutes_roll_8",
    "ict_index_roll_4", "ict_index_roll_8",
    "bps_roll_4", "bps_roll_8",
    "goals_scored_roll_4", "assists_roll_4",
    "clean_sheets_roll_4",
    "influence_roll_4", "creativity_roll_4", "threat_roll_4",
    "total_points_momentum", "minutes_momentum",
    "ict_index_momentum",
    "transfers_net",
    "saves_roll_4",
    "penalty_taker",
    "opponent_xg_for_roll_4",
]

ID_COLUMNS = ["element", "name", "position", "team", "now_cost"]

# Full feature set = base rolling features + fixture context features
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + FIXTURE_FEATURE_COLUMNS


def load_model(path: Path):
    """Load a joblib-serialized model."""
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def get_feature_columns() -> list[str]:
    """Return the list of feature columns expected by models."""
    return FEATURE_COLUMNS.copy()


def predict_next_gw(
    player_features: pd.DataFrame,
    model_path: Path = ACTIVE_MODEL,
) -> pd.DataFrame:
    """Generate xP predictions for the next gameweek."""
    model = load_model(model_path)
    feature_cols = get_feature_columns()

    # Normalize cost column: vaastav uses 'value', API uses 'now_cost'
    df = player_features.copy()
    if "now_cost" not in df.columns and "value" in df.columns:
        df["now_cost"] = df["value"]

    X = df[feature_cols].copy()
    X = X.fillna(0)

    predictions = model.predict(X)
    predictions = np.clip(predictions, 0, None)  # xP can't be negative

    result = df[ID_COLUMNS].copy()
    result["xP"] = predictions
    return result


def load_position_models(models_config: dict | None = None) -> dict:
    """Load all per-position models. Returns dict {position: model or None}.

    If a model file does not exist, returns None for that position (triggers fallback).
    models_config defaults to ACTIVE_MODELS from config.
    """
    if models_config is None:
        models_config = get_active_models()
    result = {}
    for pos, path in models_config.items():
        if path is not None and Path(path).exists():
            result[pos] = load_model(Path(path))
        else:
            result[pos] = None
    return result


def predict_next_gw_per_position(
    player_features: pd.DataFrame,
    models: dict | None = None,
    ep_next_map: dict | None = None,
) -> pd.DataFrame:
    """Generate xP using per-position models with DGW aggregation.

    player_features: DataFrame where DGW players have 2 rows (one per fixture).
    models: dict {position_str: model_or_None}. If None, loads from ACTIVE_MODELS.
    ep_next_map: {element_id: ep_next_value} fallback when model is None.

    Returns one row per player (DGW summed). Columns: element, code, name, position,
    team, now_cost, xP, _fallback (bool).
    """
    if models is None:
        models = load_position_models()

    df = player_features.copy()
    if "now_cost" not in df.columns and "value" in df.columns:
        df["now_cost"] = df["value"]

    # Normalize position to string label if stored as integer.
    # Guard: pandas 3.0 uses StringDtype (not object) for strings, so check
    # for integer dtype explicitly — not "not object" — to avoid remapping
    # already-string positions through an integer-keyed dict (→ all NaN).
    from src.pipeline.fetch import ELEMENT_TYPE_MAP
    if pd.api.types.is_integer_dtype(df["position"]):
        df["position"] = df["position"].map(ELEMENT_TYPE_MAP)

    feature_cols = ALL_FEATURE_COLUMNS
    predictions = []

    for pos, model in models.items():
        pos_df = df[df["position"] == pos].copy()
        if pos_df.empty:
            continue

        available_features = [c for c in feature_cols if c in pos_df.columns]
        X = pos_df[available_features].fillna(0)

        if model is not None and len(available_features) >= len(FEATURE_COLUMNS):
            # Predict per row to support DGW (where each fixture row gets its own prediction)
            try:
                xp_list = []
                for i in range(len(X)):
                    row_X = X.iloc[[i]]
                    row_xp = model.predict(row_X)
                    xp_list.append(float(row_xp[0]))
                xp = np.clip(xp_list, 0, None)
                pos_df = pos_df.copy()
                pos_df["xP"] = xp
                pos_df["_fallback"] = False
            except Exception as exc:
                # Handles sklearn version mismatches (e.g. model trained on 1.7, running 1.8)
                logger.warning(
                    f"[predict] {pos} model.predict() failed ({exc}) — falling back to ep_next"
                )
                if ep_next_map:
                    pos_df["xP"] = pos_df["element"].map(ep_next_map).fillna(0.0)
                else:
                    pos_df["xP"] = 0.0
                pos_df["_fallback"] = True
        else:
            # Fallback: use ep_next if provided, else 0
            if ep_next_map:
                pos_df["xP"] = pos_df["element"].map(ep_next_map).fillna(0.0)
            else:
                pos_df["xP"] = 0.0
            pos_df["_fallback"] = True

        predictions.append(pos_df)

    if not predictions:
        return pd.DataFrame(columns=ID_COLUMNS + ["xP", "_fallback"])

    combined = pd.concat(predictions, ignore_index=True)

    # Aggregate: sum xP across fixtures per player (handles DGW)
    agg_cols = {
        "xP": "sum",
        "_fallback": "first",
        "now_cost": "first",
        "team": "first",
        "position": "first",
        "name": "first",
    }
    if "code" in combined.columns:
        agg_cols["code"] = "first"
    if "raw_xP" in combined.columns:
        agg_cols["raw_xP"] = "sum"

    group_key = "element"
    result = combined.groupby(group_key, as_index=False).agg(agg_cols)
    return result


def apply_xp_corrections(
    predictions: pd.DataFrame,
    bootstrap: dict,
    target_gw: int,
) -> pd.DataFrame:
    """Apply blank GW zeroing to raw xP predictions.

    Returns dataframe with an added ``raw_xP`` column (original ML output) and a
    corrected ``xP`` column (zero for blank-GW players).  All downstream consumers
    (optimizer, recommend.py, analysis.py) should use the corrected ``xP``.

    Blank GW detection uses the bootstrap ``scout_risks`` field which FPL marks
    explicitly when a player has no fixture in the target gameweek.  This is more
    reliable than inferring blanks from the fixtures endpoint (which may not be
    archived) and correctly handles mid-season postponements.
    """
    df = predictions.copy()
    df["raw_xP"] = df["xP"].copy()

    blank_elements = {
        el["id"]
        for el in bootstrap.get("elements", [])
        for risk in el.get("scout_risks", [])
        if risk.get("property") == "blank_gw" and risk.get("gameweek") == target_gw
    }

    if blank_elements:
        df.loc[df["element"].isin(blank_elements), "xP"] = 0.0

    return df


def save_full_predictions(predictions: pd.DataFrame, path: Path) -> None:
    """Save full player predictions to CSV.

    Columns: element, code, name, position, team, xP, now_cost (in 0.1M units, e.g. 105 = £10.5m)
    now_cost is kept in 0.1M units (FPL API convention) so recommend.py and optimize.py
    can use it directly without unit conversion. Convert to £ only in user-facing output (CSVs,
    terminal summaries) by dividing by 10.
    """
    df = predictions.copy()
    # Ensure code column exists (may be absent if model ran without cross-season data)
    if "code" not in df.columns:
        df["code"] = df.get("element", pd.Series(dtype=int))
    # now_cost stays in 0.1M units — do NOT divide by 10 here
    cols = ["element", "code", "name", "position", "team", "xP", "raw_xP", "now_cost"]
    df = df[[c for c in cols if c in df.columns]]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def compute_shap_reasons(
    model,
    X: pd.DataFrame,
    feature_cols: list[str],
    top_n: int = 5,
    cohort_X: pd.DataFrame | None = None,
) -> pd.Series:
    """Return pipe-separated top-N SHAP feature contributions per player row.

    Format: "minutes_roll_4 13.2 (rank 1/860) | ict_index_roll_4 45.1 (rank 3/860) | ..."
    Rank is descending (rank 1 = highest value) within cohort_X (full position set).
    If cohort_X is None, ranks are computed within X itself.

    Raises ValueError on feature column mismatch to prevent silent wrong labels.
    """
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        if expected != feature_cols:
            raise ValueError(
                f"compute_shap_reasons column mismatch: "
                f"model expects {expected[:5]}..., got {feature_cols[:5]}..."
            )

    X_clean = X[feature_cols].fillna(0)
    cohort_clean = (cohort_X[feature_cols].fillna(0)
                    if cohort_X is not None else X_clean)
    cohort_n = len(cohort_clean)

    cohort_ranks = cohort_clean.rank(ascending=False, method="min").astype(int)
    x_ranks = cohort_ranks.reindex(X_clean.index)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_clean)  # shape (n, p) or (n, p, outputs)
    if isinstance(shap_values, list):
        # Some versions return a list of arrays for multi-output
        shap_values = shap_values[0]
    shap_arr = np.array(shap_values)
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[..., 0]

    reasons = []
    for i, row_shap in enumerate(shap_arr):
        abs_idx = np.argsort(np.abs(row_shap))[::-1][:top_n]
        row = X_clean.iloc[i]
        row_ranks = x_ranks.iloc[i]
        parts = [
            f"{feature_cols[j]} {row[feature_cols[j]]:.2f} (rank {row_ranks[feature_cols[j]]}/{cohort_n})"
            for j in abs_idx
        ]
        reasons.append(" | ".join(parts))
    return pd.Series(reasons, index=X.index)

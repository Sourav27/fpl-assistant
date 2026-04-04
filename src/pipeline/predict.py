# src/pipeline/predict.py
"""Load trained models and generate predictions."""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import ACTIVE_MODEL


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
]

ID_COLUMNS = ["element", "name", "position", "team", "now_cost"]


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

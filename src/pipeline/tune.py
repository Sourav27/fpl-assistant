"""Optuna-based hyperparameter tuning for per-position RF and XGBoost models."""
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
    feature_cols: list,
    pos: str,
    min_rows: int = 200,
) -> None:
    """Raise ValueError if training data is degenerate."""
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
        logger.warning("[tune] %s: target is constant — CV rho will be NaN", pos)


def _cv_rho_timeseries(model_fn, X: pd.DataFrame, y: pd.Series) -> float:
    """Mean Spearman rho across TimeSeriesSplit folds."""
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
    feat_cols: list,
    algos: list | None = None,
    n_trials: int = 50,
) -> tuple:
    """Tune RF and/or XGBoost for one position using Optuna TPE + TimeSeriesSplit CV.

    X_train must be sorted by (season, GW) — caller is responsible.

    Returns: (best_model, best_algo, best_params, best_cv_rho)
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
        print(f"[tune] {pos}/{algo.upper()} best CV rho={trial_rho:.4f} params={trial_params}")

        if not np.isnan(trial_rho) and trial_rho > best_rho:
            best_rho = trial_rho
            best_algo = algo
            best_params = trial_params

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

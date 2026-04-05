"""xG source validation gate for Track H / Track B dependency.

Gate rule: understat_rho >= fpl_opta_rho - tolerance → use understat xGC.
If gate fails → fall back to vaastav goals_conceded for xGC_rolling_4.

Usage:
    python -m src.pipeline.source_validation
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
import pandas as pd
from scipy.stats import spearmanr
from src.config import SOURCE_VALIDATION_CSV

logger = logging.getLogger(__name__)


@dataclass
class SourceValidationResult:
    passed: bool
    understat_rho: float
    fpl_opta_rho: float
    recommended_source: str   # "understat" | "vaastav_goals_conceded"
    n_samples: int = 0


def compute_source_spearman(df: pd.DataFrame, xg_col: str, actual_col: str) -> float:
    """Compute Spearman ρ between an xG column and actual goals column."""
    clean = df[[xg_col, actual_col]].dropna()
    if len(clean) < 10:
        logger.warning("Only %d samples for Spearman ρ — result unreliable", len(clean))
    rho, _ = spearmanr(clean[xg_col], clean[actual_col])
    return float(rho)


def run_xg_validation_gate(
    understat_rho: float,
    fpl_opta_rho: float,
    tolerance: float = 0.05,
    n_samples: int = 0,
) -> SourceValidationResult:
    """Evaluate whether understat xG is reliable enough to use in Track B.

    Gate: understat_rho >= fpl_opta_rho - tolerance → use understat.
    If failed: fall back to vaastav goals_conceded aggregation.
    """
    passed = understat_rho >= (fpl_opta_rho - tolerance)
    source = "understat" if passed else "vaastav_goals_conceded"
    return SourceValidationResult(
        passed=passed,
        understat_rho=understat_rho,
        fpl_opta_rho=fpl_opta_rho,
        recommended_source=source,
        n_samples=n_samples,
    )


def append_validation_result(result: SourceValidationResult, run_date: str) -> None:
    """Append a validation result row to results/source_validation.csv."""
    row = pd.DataFrame([{
        "run_date": run_date,
        "understat_rho": result.understat_rho,
        "fpl_opta_rho": result.fpl_opta_rho,
        "n_samples": result.n_samples,
        "gate_passed": result.passed,
        "recommended_source": result.recommended_source,
    }])
    if SOURCE_VALIDATION_CSV.exists():
        row.to_csv(SOURCE_VALIDATION_CSV, mode="a", header=False, index=False)
    else:
        SOURCE_VALIDATION_CSV.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(SOURCE_VALIDATION_CSV, index=False)
    logger.info("Validation result appended to %s", SOURCE_VALIDATION_CSV)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Run gate manually — provide understat_rho and fpl_opta_rho as arguments.")

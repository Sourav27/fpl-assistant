import pandas as pd
import pytest


@pytest.fixture
def sample_accuracy_log(tmp_path):
    df = pd.DataFrame([
        {"gw": 31, "season": "2025-26", "your_pts": 44, "wildcard_pts": 54, "recommended_pts": 8,
         "your_percentile_rank": 20, "best_score": 109, "avg_score": 38},
        {"gw": 32, "season": "2025-26", "your_pts": 54, "wildcard_pts": 38, "recommended_pts": 7,
         "your_percentile_rank": 30, "best_score": 132, "avg_score": 46},
    ])
    p = tmp_path / "accuracy_log.csv"
    df.to_csv(p, index=False)
    return p


def test_load_accuracy_log_sorted_by_season_gw(sample_accuracy_log):
    from scripts.generate_reports import load_accuracy_log
    df = load_accuracy_log(sample_accuracy_log, from_gw=31)
    assert list(df["gw"]) == [31, 32]


def test_load_accuracy_log_filters_from_gw(sample_accuracy_log):
    from scripts.generate_reports import load_accuracy_log
    df = load_accuracy_log(sample_accuracy_log, from_gw=32)
    assert list(df["gw"]) == [32]


def test_estimate_rank_percentile_midpoint():
    from scripts.generate_reports import estimate_rank_percentile
    pct = estimate_rank_percentile(score=73, best_score=109, avg_score=38)
    assert 0.001 < pct < 50.0


def test_estimate_rank_percentile_at_avg():
    from scripts.generate_reports import estimate_rank_percentile
    pct = estimate_rank_percentile(score=38, best_score=109, avg_score=38)
    assert abs(pct - 50.0) < 0.1


def test_estimate_rank_percentile_above_best():
    from scripts.generate_reports import estimate_rank_percentile
    pct = estimate_rank_percentile(score=150, best_score=109, avg_score=38)
    assert pct <= 0.001

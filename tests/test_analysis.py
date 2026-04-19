import pytest
import pandas as pd
from src.pipeline.analysis import (
    compute_prediction_misses,
    compute_dream_team,
    format_post_match_summary,
    append_accuracy_log,
)


class TestComputePredictionMisses:
    def test_identifies_overperformer(self):
        picks_df = pd.DataFrame({
            "element": [1, 2, 3],
            "name": ["Saka", "Haaland", "Palmer"],
            "xP": [6.8, 8.5, 4.2],
            "actual_points": [12, 2, 12],
        })
        misses = compute_prediction_misses(picks_df)
        # Palmer: +7.8 (actual - xP), Saka: +5.2, Haaland: -6.5
        names = [m["name"] for m in misses]
        assert "Haaland" in names
        assert "Palmer" in names
        # Sorted by abs(miss) descending
        assert abs(misses[0]["miss"]) >= abs(misses[1]["miss"])

    def test_miss_is_actual_minus_predicted(self):
        picks_df = pd.DataFrame({
            "element": [1],
            "name": ["Haaland"],
            "xP": [8.5],
            "actual_points": [2],
        })
        misses = compute_prediction_misses(picks_df)
        assert misses[0]["miss"] == pytest.approx(2 - 8.5)


class TestComputeDreamTeam:
    def test_selects_highest_scoring_xi(self):
        live_data = pd.DataFrame({
            "element": range(1, 26),
            "name": [f"P{i}" for i in range(1, 26)],
            "position": (["GK"] * 2 + ["DEF"] * 6 + ["MID"] * 8 + ["FWD"] * 9),
            "total_points": [i * 2 for i in range(1, 26)],
            "team": [f"T{i % 8}" for i in range(1, 26)],
        })
        dream = compute_dream_team(live_data)
        assert len(dream) == 11
        # All elements in dream are from the original data
        assert all(e in live_data["element"].values for e in dream["element"])

    def test_dream_team_valid_formation(self):
        live_data = pd.DataFrame({
            "element": range(1, 26),
            "name": [f"P{i}" for i in range(1, 26)],
            "position": (["GK"] * 2 + ["DEF"] * 6 + ["MID"] * 8 + ["FWD"] * 9),
            "total_points": [i * 2 for i in range(1, 26)],
            "team": [f"T{i % 8}" for i in range(1, 26)],
        })
        dream = compute_dream_team(live_data)
        pos_counts = dream["position"].value_counts()
        assert pos_counts.get("GK", 0) == 1
        assert pos_counts.get("DEF", 0) >= 3
        assert pos_counts.get("MID", 0) >= 2
        assert pos_counts.get("FWD", 0) >= 1


class TestAppendAccuracyLog:
    def test_creates_log_on_first_run(self, tmp_path):
        log_path = tmp_path / "accuracy_log.csv"
        append_accuracy_log(
            path=log_path,
            gw=33,
            your_pts=58, your_xp=72.3,
            recommended_pts=65, recommended_xp=78.1,
            dream_pts=89,
            your_percentile_rank=20,
            benchmarks={"best_score": 109, "top_1k_score": 85,
                        "top_10k_score": 79, "top_100k_score": 73,
                        "top_1m_score": 62, "avg_score": 38, "median_score": 36},
            ranked_count=12914049,
        )
        df = pd.read_csv(log_path)
        assert len(df) == 1
        assert df.iloc[0]["gw"] == 33
        assert df.iloc[0]["your_pts"] == 58

    def test_appends_to_existing_log(self, tmp_path):
        log_path = tmp_path / "accuracy_log.csv"
        kwargs = dict(your_pts=60, your_xp=65.0, recommended_pts=None, recommended_xp=None,
                      dream_pts=None, your_percentile_rank=None, benchmarks={}, ranked_count=0)
        append_accuracy_log(log_path, gw=31, **kwargs)
        append_accuracy_log(log_path, gw=32, **kwargs)
        df = pd.read_csv(log_path)
        assert len(df) == 2
        assert list(df["gw"]) == [31, 32]


class TestSpearmanRho:
    """Tests for compute_spearman_rho() — B-F7."""

    def test_perfect_rank_correlation(self):
        from src.pipeline.analysis import compute_spearman_rho
        import pandas as pd
        df = pd.DataFrame({
            "xP": [1.0, 2.0, 3.0, 4.0, 5.0],
            "actual_points": [2, 4, 6, 8, 10],
        })
        rho = compute_spearman_rho(df)
        assert rho == pytest.approx(1.0, abs=0.01)

    def test_inverse_rank_correlation(self):
        from src.pipeline.analysis import compute_spearman_rho
        import pandas as pd
        df = pd.DataFrame({
            "xP": [5.0, 4.0, 3.0, 2.0, 1.0],
            "actual_points": [1, 2, 3, 4, 5],
        })
        rho = compute_spearman_rho(df)
        assert rho == pytest.approx(-1.0, abs=0.01)

    def test_returns_nan_for_empty(self):
        from src.pipeline.analysis import compute_spearman_rho
        import pandas as pd
        rho = compute_spearman_rho(pd.DataFrame({"xP": [], "actual_points": []}))
        import math
        assert math.isnan(rho)

    def test_accuracy_log_has_spearman_rho_column(self, tmp_path):
        from src.pipeline.analysis import append_accuracy_log
        import pandas as pd
        log_path = tmp_path / "accuracy_log.csv"
        picks = pd.DataFrame({
            "xP": [3.0, 7.0, 2.0],
            "raw_xP": [3.0, 7.0, 2.0],
            "actual_points": [4, 8, 1],
        })
        append_accuracy_log(
            gw=32,
            your_pts=42,
            your_xp=38.0,
            recommended_pts=None,
            recommended_xp=None,
            wildcard_pts=None,
            wildcard_xp=None,
            dream_team_pts=None,
            benchmarks={"average": 40, "top_player": 60},
            your_percentile_rank=None,
            picks_df=picks,
            path=log_path,
        )
        log = pd.read_csv(log_path)
        assert "spearman_rho" in log.columns
        assert not pd.isna(log.iloc[0]["spearman_rho"])


def test_append_accuracy_log_writes_season_column(tmp_path):
    log = tmp_path / "accuracy_log.csv"
    append_accuracy_log(log, gw=31, season="2025-26",
                        your_pts=44, your_xp=44.3,
                        recommended_pts=8, recommended_xp=38.8)
    import pandas as pd
    df = pd.read_csv(log)
    assert "season" in df.columns
    assert df.iloc[0]["season"] == "2025-26"

def test_append_accuracy_log_season_defaults_to_current_season(tmp_path):
    from src.config import CURRENT_SEASON
    log = tmp_path / "accuracy_log.csv"
    append_accuracy_log(log, gw=31, your_pts=44, your_xp=44.3,
                        recommended_pts=8, recommended_xp=38.8)
    import pandas as pd
    df = pd.read_csv(log)
    assert df.iloc[0]["season"] == CURRENT_SEASON

def test_build_actual_squad_csv_columns():
    from src.pipeline.analysis import build_actual_squad_csv
    entry_picks = {
        "picks": [
            {"element": 1, "position": 1, "multiplier": 2, "is_captain": True, "is_vice_captain": False},
            {"element": 2, "position": 12, "multiplier": 0, "is_captain": False, "is_vice_captain": True},
        ]
    }
    bootstrap = {
        "elements": [
            {"id": 1, "web_name": "Salah", "element_type": 3, "team": 14, "now_cost": 130},
            {"id": 2, "web_name": "Saka",  "element_type": 3, "team": 1,  "now_cost": 100},
        ],
        "teams": [{"id": 1, "name": "Arsenal"}, {"id": 14, "name": "Liverpool"}],
        "element_types": [
            {"id": 3, "singular_name_short": "MID"},
        ],
    }
    actual_pts = {1: 20, 2: 6}
    df = build_actual_squad_csv(entry_picks, bootstrap, actual_pts)
    assert list(df.columns) == [
        "element", "name", "position", "team", "actual_pts",
        "is_starter", "bench_order", "is_captain", "is_vice_captain", "now_cost"
    ]
    salah = df[df["element"] == 1].iloc[0]
    assert salah["actual_pts"] == 20
    assert salah["is_captain"] is True
    assert salah["is_starter"] is True
    saka = df[df["element"] == 2].iloc[0]
    assert saka["is_starter"] is False
    assert saka["bench_order"] == 1

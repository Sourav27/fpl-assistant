def test_append_accuracy_log_spearman_rho_written(tmp_path):
    """Passing picks_df to append_accuracy_log must populate spearman_rho."""
    import pandas as pd
    from src.pipeline.analysis import append_accuracy_log

    picks = pd.DataFrame({
        "element": [1, 2, 3, 4, 5],
        "name": ["A", "B", "C", "D", "E"],
        "xP": [5.0, 4.0, 3.0, 2.0, 1.0],
        "actual_points": [10, 8, 5, 4, 2],
    })
    log = tmp_path / "accuracy_log.csv"
    append_accuracy_log(
        path=log, gw=99,
        your_pts=29, your_xp=15.0,
        recommended_pts=None, recommended_xp=None,
        picks_df=picks,
    )
    df = pd.read_csv(log)
    assert df["spearman_rho"].notna().all(), "spearman_rho should be written when picks_df provided"
    assert abs(df.iloc[0]["spearman_rho"]) > 0.5

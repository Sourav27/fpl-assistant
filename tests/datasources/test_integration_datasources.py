"""Integration tests for Track H datasources.

HTTP calls are mocked via unittest.mock (not VCR) to avoid cassette files.
These tests verify the end-to-end flow from raw API response to PlayerSignal.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.pipeline.datasources.ffs import parse_ffs_feed
from src.pipeline.datasources.reddit import parse_reddit_posts
from src.pipeline.datasources.premierinjuries import parse_premierinjuries_html, cross_verify_against_fpl
from src.pipeline.datasources.signals import PlayerSignal
from src.pipeline.signal_feedback import append_signal_feedback, compute_source_accuracy
from src.pipeline.source_validation import run_xg_validation_gate, SourceValidationResult, compute_source_spearman

MINIMAL_BOOTSTRAP = {
    "elements": [
        {"id": 1, "code": 80201, "web_name": "Salah",
         "first_name": "Mohamed", "second_name": "Salah", "team": 14},
    ]
}

FFS_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Mohamed Salah doubt for GW32</title>
    <description>Salah is a doubt.</description>
    <pubDate>Sat, 05 Apr 2026 08:00:00 +0000</pubDate>
  </item>
</channel></rss>"""

REDDIT_DATA = {"data": {"children": [
    {"data": {"title": "Salah doubt for GW32", "selftext": "",
              "created_utc": 1743840000, "score": 150}}
]}}

PI_HTML = """<html><body>
<table id="player-injury-table">
  <thead><tr><th>Player</th><th>Club</th><th>Status</th><th>Notes</th></tr></thead>
  <tbody>
    <tr><td>Mohamed Salah</td><td>Liverpool</td><td>Doubt</td><td>Training knock</td></tr>
  </tbody>
</table></body></html>"""


def test_ffs_to_signal_full_chain():
    signals = parse_ffs_feed(rss_content=FFS_RSS, bootstrap_data=MINIMAL_BOOTSTRAP)
    assert len(signals) == 1
    assert signals[0].player_code == 80201
    assert signals[0].signal_type == "doubt"
    assert signals[0].source == "ffs"


def test_reddit_to_signal_full_chain():
    signals = parse_reddit_posts(posts_data=REDDIT_DATA, bootstrap_data=MINIMAL_BOOTSTRAP, min_score=50)
    assert len(signals) == 1
    assert signals[0].player_code == 80201
    assert signals[0].source == "reddit"
    assert signals[0].confidence <= 0.6


def test_pi_to_signal_full_chain():
    signals = parse_premierinjuries_html(html_content=PI_HTML, bootstrap_data=MINIMAL_BOOTSTRAP)
    assert len(signals) == 1
    assert signals[0].player_code == 80201
    assert signals[0].signal_type == "doubt"


def test_fpl_contradiction_flags_signal():
    signals = parse_premierinjuries_html(html_content=PI_HTML, bootstrap_data=MINIMAL_BOOTSTRAP)
    fpl_status = {80201: "a"}
    verified = cross_verify_against_fpl(signals, fpl_status)
    assert verified[0]["contradicted"] is True


def test_fpl_consistent_no_contradiction():
    signals = parse_premierinjuries_html(html_content=PI_HTML, bootstrap_data=MINIMAL_BOOTSTRAP)
    fpl_status = {80201: "d"}
    verified = cross_verify_against_fpl(signals, fpl_status)
    assert verified[0]["contradicted"] is False


def test_signal_feedback_full_chain(tmp_path):
    csv_path = tmp_path / "signal_accuracy.csv"
    signals = parse_ffs_feed(rss_content=FFS_RSS, bootstrap_data=MINIMAL_BOOTSTRAP)
    for sig in signals:
        append_signal_feedback(sig, gw=32, actual_started=False,
                               contradicted=False, csv_path=csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["source"] == "ffs"


def test_source_accuracy_computation(tmp_path):
    csv_path = tmp_path / "signal_accuracy.csv"
    signals = parse_ffs_feed(rss_content=FFS_RSS, bootstrap_data=MINIMAL_BOOTSTRAP)
    sig = signals[0]
    for gw, started in [(30, False), (31, False), (32, False), (33, True)]:
        append_signal_feedback(sig, gw=gw, actual_started=started,
                               contradicted=False, csv_path=csv_path)
    df = pd.read_csv(csv_path)
    acc = compute_source_accuracy(df)
    ffs_row = acc[(acc["source"] == "ffs") & (acc["signal_type"] == "doubt")]
    assert pytest.approx(ffs_row["accuracy"].values[0], abs=0.01) == 0.75


def test_xg_validation_gate_integration():
    result = run_xg_validation_gate(
        understat_rho=0.63, fpl_opta_rho=0.66, tolerance=0.05
    )
    assert isinstance(result, SourceValidationResult)
    assert result.passed is True
    assert result.recommended_source == "understat"


def test_xg_gate_spearman_both_sources():
    """Verify compute_source_spearman is callable on realistic data and gate is deterministic."""
    rng = np.random.default_rng(42)
    n = 200
    actual_goals = rng.integers(0, 3, size=n).astype(float)
    understat_xg = actual_goals + rng.normal(0, 0.3, size=n)
    fpl_opta_xg  = actual_goals + rng.normal(0, 0.35, size=n)
    df = pd.DataFrame({
        "understat_xG": understat_xg,
        "fpl_opta_xG": fpl_opta_xg,
        "goals_scored": actual_goals,
    })
    rho_u = compute_source_spearman(df, xg_col="understat_xG", actual_col="goals_scored")
    rho_f = compute_source_spearman(df, xg_col="fpl_opta_xG", actual_col="goals_scored")
    assert rho_u > 0.80
    assert rho_f > 0.75
    gate1 = run_xg_validation_gate(rho_u, rho_f, tolerance=0.05)
    gate2 = run_xg_validation_gate(rho_u, rho_f, tolerance=0.05)
    assert gate1.passed == gate2.passed
    assert gate1.recommended_source == gate2.recommended_source

import pytest
import pandas as pd
from pathlib import Path
from src.pipeline.signal_feedback import (
    append_signal_feedback,
    compute_source_accuracy,
    make_signal_id,
)
from src.pipeline.datasources.signals import PlayerSignal


def _make_signal(source="ffs", signal_type="doubt", player_code=80201):
    return PlayerSignal(
        player_code=player_code, source=source,
        signal_type=signal_type, text="test", timestamp="2026-04-05T00:00:00Z",
    )


def test_make_signal_id_deterministic():
    id1 = make_signal_id(source="ffs", player_code=80201, gw=32, signal_type="doubt")
    id2 = make_signal_id(source="ffs", player_code=80201, gw=32, signal_type="doubt")
    assert id1 == id2


def test_make_signal_id_differs_by_source():
    id1 = make_signal_id("ffs", 80201, 32, "doubt")
    id2 = make_signal_id("reddit", 80201, 32, "doubt")
    assert id1 != id2


def test_append_signal_feedback_creates_file(tmp_path):
    csv_path = tmp_path / "signal_accuracy.csv"
    sig = _make_signal()
    append_signal_feedback(
        signal=sig, gw=32, actual_started=True,
        contradicted=False, csv_path=csv_path,
    )
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["source"] == "ffs"
    assert df.iloc[0]["actual_started"] == True


def test_append_signal_feedback_appends(tmp_path):
    csv_path = tmp_path / "signal_accuracy.csv"
    sig = _make_signal()
    append_signal_feedback(sig, gw=32, actual_started=True,
                           contradicted=False, csv_path=csv_path)
    append_signal_feedback(sig, gw=33, actual_started=False,
                           contradicted=False, csv_path=csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 2


def test_compute_source_accuracy():
    df = pd.DataFrame([
        {"source": "ffs", "signal_type": "doubt", "actual_started": False},
        {"source": "ffs", "signal_type": "doubt", "actual_started": False},
        {"source": "ffs", "signal_type": "doubt", "actual_started": True},
        {"source": "reddit", "signal_type": "doubt", "actual_started": True},
    ])
    acc = compute_source_accuracy(df)
    ffs_acc = acc.loc[(acc["source"] == "ffs") & (acc["signal_type"] == "doubt"), "accuracy"]
    assert pytest.approx(ffs_acc.values[0], abs=0.01) == 2/3


def test_signal_feedback_schema(tmp_path):
    csv_path = tmp_path / "signal_accuracy.csv"
    sig = _make_signal()
    append_signal_feedback(sig, gw=32, actual_started=True,
                           contradicted=False, csv_path=csv_path)
    df = pd.read_csv(csv_path)
    required_cols = {
        "signal_id", "source", "signal_type", "player_code",
        "gw", "predicted_status", "actual_started", "contradicted", "run_date"
    }
    assert required_cols.issubset(set(df.columns))

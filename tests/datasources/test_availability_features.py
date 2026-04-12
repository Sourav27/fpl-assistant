"""Tests for availability_features.py — feature assembly layer.

Contract verified:
  H-C6-T1  status='i' → is_injured=1, is_suspended=0, is_doubt=0
  H-C6-T2  status='u' → is_injured=1
  H-C6-T3  status='s' → is_suspended=1, is_injured=0, is_doubt=0
  H-C6-T4  status='d' → is_doubt=1, is_injured=0, is_suspended=0
  H-C6-T5  status='d', chance=100 → is_doubt=1 (chance does NOT clear is_doubt)
  H-C6-T6  status='a', no news → premierinjuries fallback triggers
  H-C6-T7  status='a', has news → premierinjuries fallback NOT triggered
  H-C6-T8  status='a', no news, pi=None → no features set
  H-C6-T9  corroborating sources increment n_corroborating_sources
  H-C6-T10 signal_confidence is 1.0 when only FPL signals and no corroboration
  H-C6-T11 signal_confidence rises with agreeing corroborating sources
  H-C6-T12 status='n' → all features 0 (not-in-squad is not an injury)
  H-C6-T13 status='a', healthy → signal_confidence=0.0
"""
from src.pipeline.datasources.availability_features import (
    AvailabilityFeatures,
    AvailabilitySignal,
    compute_availability_features,
)


def test_injured_status():
    """H-C6-T1: status='i' → is_injured=1, others 0."""
    feat = compute_availability_features("i", "Knee injury")
    assert feat.is_injured == 1
    assert feat.is_suspended == 0
    assert feat.is_doubt == 0


def test_unavailable_status():
    """H-C6-T2: status='u' → is_injured=1."""
    feat = compute_availability_features("u", "Illness")
    assert feat.is_injured == 1


def test_suspended_status():
    """H-C6-T3: status='s' → is_suspended=1, is_injured=0."""
    feat = compute_availability_features("s", "Suspension")
    assert feat.is_suspended == 1
    assert feat.is_injured == 0
    assert feat.is_doubt == 0


def test_doubt_status():
    """H-C6-T4: status='d' → is_doubt=1."""
    feat = compute_availability_features("d", "Knock - 75% chance")
    assert feat.is_doubt == 1
    assert feat.is_injured == 0
    assert feat.is_suspended == 0


def test_doubt_with_chance_100_still_is_doubt():
    """H-C6-T5: status='d' with chance=100 — is_doubt driven by status, not chance."""
    feat = compute_availability_features("d", "Doubtful (100%)")
    assert feat.is_doubt == 1


def test_premierinjuries_fallback_when_status_a_no_news():
    """H-C6-T6: status='a', no FPL news → premierinjuries fallback activates."""
    feat = compute_availability_features("a", "", premierinjuries_status="injured")
    assert feat.is_injured == 1
    assert feat.is_doubt == 0


def test_premierinjuries_fallback_suppressed_when_fpl_has_news():
    """H-C6-T7: status='a' but FPL news present → premierinjuries ignored."""
    feat = compute_availability_features("a", "Hamstring - day to day", premierinjuries_status="injured")
    # FPL says 'a' (available), news is present — premierinjuries should NOT override
    assert feat.is_injured == 0


def test_no_signal_when_premierinjuries_is_none():
    """H-C6-T8: status='a', no news, pi=None → everything 0."""
    feat = compute_availability_features("a", "", premierinjuries_status=None)
    assert feat.is_injured == 0
    assert feat.is_doubt == 0
    assert feat.signal_confidence == 0.0


def test_corroboration_increments_count():
    """H-C6-T9: FFS and Reddit agreeing → n_corroborating_sources=2."""
    sigs = [
        AvailabilitySignal(source="ffs", is_injured=True),
        AvailabilitySignal(source="reddit", is_doubt=True),
    ]
    feat = compute_availability_features("i", "Injury", secondary_signals=sigs)
    assert feat.n_corroborating_sources == 2


def test_signal_confidence_fpl_only():
    """H-C6-T10: only FPL signal → confidence = 1.0 (weight = 1.0 / 1 source)."""
    feat = compute_availability_features("i", "Muscle injury")
    assert feat.signal_confidence == 1.0


def test_signal_confidence_rises_with_corroboration():
    """H-C6-T11: FPL (1.0) + FFS (0.6) → confidence = (1.0+0.6)/2 = 0.8."""
    sigs = [AvailabilitySignal(source="ffs", is_injured=True)]
    feat = compute_availability_features("i", "Knock", secondary_signals=sigs)
    assert abs(feat.signal_confidence - 0.8) < 1e-6


def test_not_in_squad_all_zeros():
    """H-C6-T12: status='n' (not in squad) → all feature flags 0."""
    feat = compute_availability_features("n", "")
    assert feat.is_injured == 0
    assert feat.is_suspended == 0
    assert feat.is_doubt == 0


def test_available_player_confidence_zero():
    """H-C6-T13: healthy available player → signal_confidence=0.0."""
    feat = compute_availability_features("a", "")
    assert feat.signal_confidence == 0.0
    assert feat.is_injured == 0
    assert feat.is_doubt == 0
    assert feat.n_corroborating_sources == 0

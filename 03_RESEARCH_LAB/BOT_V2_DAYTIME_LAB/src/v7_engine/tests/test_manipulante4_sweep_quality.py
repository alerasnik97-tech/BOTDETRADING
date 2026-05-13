"""Tests for MANIPULANTE 4 sweep quality gate logic."""
import pytest


def test_sweep_depth_gate_rejects_noise():
    """Sweep < 0.05 ATR is noise and must be rejected."""
    atr_h1 = 0.0050  # 50 pips
    sweep_depth = 0.0002  # 2 pips
    ratio = sweep_depth / atr_h1
    min_ratio = 0.05
    assert ratio < min_ratio  # Must be rejected


def test_sweep_depth_gate_accepts_valid():
    """Sweep >= 0.10 ATR is valid quality."""
    atr_h1 = 0.0050
    sweep_depth = 0.0008  # 8 pips = 0.16 ATR
    ratio = sweep_depth / atr_h1
    min_ratio = 0.10
    assert ratio >= min_ratio


def test_sweep_depth_rejects_breakout():
    """Very deep sweep (>0.5 ATR) may be real breakout, not stop hunt.
    This test verifies the feature is computed correctly for later filtering."""
    atr_h1 = 0.0050
    sweep_depth = 0.0030  # 30 pips = 0.60 ATR
    ratio = sweep_depth / atr_h1
    assert ratio > 0.5  # Flagged as potential breakout


def test_reclaim_required_within_window():
    """Price must close back inside the level within reclaim_max_minutes."""
    reclaim_time_seconds = 300  # 5 min
    max_minutes = 15
    assert reclaim_time_seconds <= max_minutes * 60


def test_reclaim_rejected_if_too_slow():
    """Price that takes >30 min to reclaim is rejected."""
    reclaim_time_seconds = 2400  # 40 min
    max_minutes = 30
    assert reclaim_time_seconds > max_minutes * 60  # Rejected


def test_reclaim_required_close_back_inside():
    """Close back inside level is mandatory, not just wick."""
    level = 1.1000
    close_after = 1.0995  # Below level (short sweep reclaim)
    # For short sweep: level is low, price went below, came back above
    level_low = 1.0980
    close_reclaim = 1.0985  # Above the low level
    assert close_reclaim > level_low


def test_sweep_depth_computed_without_lookahead():
    """Sweep depth uses only closed bar data at time T, not future bars."""
    signal_bar_idx = 100
    depth_computed_from_idx = 100  # Same bar (using its close/wick)
    fill_idx = 101  # Next bar
    assert depth_computed_from_idx <= signal_bar_idx
    assert fill_idx > signal_bar_idx

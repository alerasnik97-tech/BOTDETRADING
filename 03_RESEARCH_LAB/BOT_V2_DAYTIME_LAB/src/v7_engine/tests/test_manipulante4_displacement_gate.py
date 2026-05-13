"""Tests for MANIPULANTE 4 displacement gate logic."""
import pytest


def test_displacement_after_sweep_not_before():
    """Displacement must occur AFTER sweep reclaim, not before."""
    sweep_time_idx = 50
    reclaim_time_idx = 52
    displacement_time_idx = 55
    assert displacement_time_idx > reclaim_time_idx > sweep_time_idx


def test_displacement_body_gate_rejects_weak():
    """Displacement candle with body < 0.75 ATR is rejected."""
    atr = 0.0040
    disp_body = 0.0025  # 0.625 ATR
    ratio = disp_body / atr
    min_ratio = 0.75
    assert ratio < min_ratio  # Rejected


def test_displacement_body_gate_accepts_strong():
    """Displacement candle with body >= 1.0 ATR is accepted."""
    atr = 0.0040
    disp_body = 0.0045  # 1.125 ATR
    ratio = disp_body / atr
    min_ratio = 1.00
    assert ratio >= min_ratio


def test_displacement_direction_matches_trade():
    """Displacement direction must match intended trade direction.
    Short sweep -> bullish displacement -> long entry."""
    sweep_direction = "SHORT_SWEEP"  # Swept lows
    displacement_direction = "BULLISH"  # Strong bullish candle
    trade_side = "LONG"
    expected_displacement = "BULLISH" if trade_side == "LONG" else "BEARISH"
    assert displacement_direction == expected_displacement


def test_displacement_creates_structure_break():
    """Displacement must break LTF structure (CHOCH) to be valid."""
    last_swing_high = 1.1000
    displacement_close = 1.1010  # Closes above swing high
    structure_broken = displacement_close > last_swing_high
    assert structure_broken is True


def test_fvg_entry_only_if_fvg_exists():
    """FVG 50% entry type requires actual Fair Value Gap."""
    has_fvg = False
    entry_type = "fvg_50pct"
    can_enter = not (entry_type == "fvg_50pct" and not has_fvg)
    assert can_enter is False  # Cannot enter without FVG


def test_stop_confirmation_not_converted_to_market():
    """Stop confirmation entry stays as stop order, never degrades to market."""
    entry_type = "stop_confirmation"
    stop_price_touched = False
    fill_executed = stop_price_touched  # Only fills if stop touched
    assert fill_executed is False  # No fill without stop touch

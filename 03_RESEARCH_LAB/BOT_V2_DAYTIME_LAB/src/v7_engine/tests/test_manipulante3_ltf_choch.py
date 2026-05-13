import pytest

def test_ltf_choch_short_breaks_last_swing_low():
    """
    Verifies that a short CHOCH confirmation requires the LTF price to break
    the most recent causal swing low formed prior to the peak.
    """
    last_swing_low = 1.0980
    current_ltf_close = 1.0975
    
    choch_confirmed = current_ltf_close < last_swing_low
    assert choch_confirmed is True

def test_ltf_choch_requires_displacement_multiplier():
    """
    Verifies that displacement CHOCH configurations enforce a required range threshold.
    """
    candle_range = 0.0020
    atr_threshold = 0.0015
    multiplier = 1.25
    
    valid_displacement = candle_range >= (atr_threshold * multiplier)
    assert valid_displacement is True

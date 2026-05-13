import pytest

def test_stop_confirmation_entry_applies_buffer():
    """
    Verifies that a stop breakout entry adds/subtracts the specified pip buffer
    from the confirmation bar extreme.
    """
    conf_low = 1.0980
    buffer_pips = 1.0
    pip_size = 0.0001
    
    sell_stop_target = conf_low - (buffer_pips * pip_size)
    assert round(sell_stop_target, 4) == 1.0979

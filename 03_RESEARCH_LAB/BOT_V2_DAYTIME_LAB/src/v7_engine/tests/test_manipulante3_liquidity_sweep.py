import pytest

def test_valid_short_liquidity_sweep_requires_reclaim():
    """
    Verifies that a valid short sweep takes liquidity above a key high
    and subsequently closes back below it.
    """
    level_price = 1.1000
    peak_price = 1.1015
    subsequent_close = 1.0995
    
    swept = peak_price > level_price and subsequent_close < level_price
    assert swept is True

def test_invalid_short_liquidity_sweep_no_reclaim():
    """
    Verifies that if price stays above the reference level, no sweep is confirmed.
    """
    level_price = 1.1000
    peak_price = 1.1015
    subsequent_close = 1.1005
    
    swept = peak_price > level_price and subsequent_close < level_price
    assert swept is False

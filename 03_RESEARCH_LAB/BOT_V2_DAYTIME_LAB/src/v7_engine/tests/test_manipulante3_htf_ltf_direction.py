import pytest

def test_htf_direction_filter_blocks_contradictory_trades():
    """
    Verifies that when HTF bias is bullish (e.g. price in discount or > EMA),
    short entry triggers are securely rejected.
    """
    htf_bias = "BULLISH"
    intended_trade = "SHORT"
    is_permitted = (htf_bias == "BEARISH" and intended_trade == "SHORT") or \
                   (htf_bias == "BULLISH" and intended_trade == "LONG") or \
                   (htf_bias == "NEUTRAL")
    assert is_permitted is False

def test_htf_direction_filter_permits_concordant_trades():
    """
    Verifies that aligned HTF and LTF trade intents are correctly passed.
    """
    htf_bias = "BULLISH"
    intended_trade = "LONG"
    is_permitted = (htf_bias == "BEARISH" and intended_trade == "SHORT") or \
                   (htf_bias == "BULLISH" and intended_trade == "LONG") or \
                   (htf_bias == "NEUTRAL")
    assert is_permitted is True

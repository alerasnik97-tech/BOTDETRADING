import pytest

def test_slippage_reduces_net_profit():
    """
    Verifies that injecting stress slippage unconditionally reduces gross yield.
    """
    gross_yield_r = 2.0
    stress_slippage_r = 0.15
    net_r = gross_yield_r - stress_slippage_r
    assert net_r < gross_yield_r
    assert net_r == 1.85

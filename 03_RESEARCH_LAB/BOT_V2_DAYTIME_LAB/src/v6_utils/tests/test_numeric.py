
import pytest
from v6_utils.numeric import snap_to_tick, safe_add, pip_to_price

def test_snap_basic():
    # snap_to_tick(1.0245678) == 1.02457
    assert snap_to_tick(1.0245678) == 1.02457

def test_snap_no_drift():
    # 10000 sumas y restas de 0.00001
    price = 1.10000
    for _ in range(10000):
        price = safe_add(price, 0.00001)
    
    # 1.10000 + 10000 * 0.00001 = 1.10000 + 0.1 = 1.2
    assert abs(price - 1.20000) < 1e-10

def test_banker_rounding_avoided():
    # 1.024565 -> 1.02457 (ROUND_HALF_UP)
    # round(1.024565, 5) en Python 3 daría 1.02456 (Banker's rounding)
    assert snap_to_tick(1.024565) == 1.02457

def test_safe_add_commutative():
    a, b = 1.123456, 0.000014
    assert safe_add(a, b) == safe_add(b, a)

def test_pip_to_price():
    # 10 pips = 0.00100
    assert pip_to_price(10) == 0.001
    # 1.5 pips = 0.00015
    assert pip_to_price(1.5) == 0.00015

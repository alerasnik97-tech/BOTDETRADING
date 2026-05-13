
from __future__ import annotations
from decimal import Decimal, ROUND_HALF_UP
import math

TICK_SIZE = {
    "EURUSD": Decimal("0.00001")
}

def snap_to_tick(price: float, instrument: str = "EURUSD") -> float:
    """
    Ajusta el precio al múltiplo más cercano del tick size usando ROUND_HALF_UP.
    Evita el Banker's Rounding de Python.
    """
    if instrument not in TICK_SIZE:
        raise ValueError(f"Instrumento {instrument} no soportado en TICK_SIZE.")
        
    tick = TICK_SIZE[instrument]
    # Determinar decimales del tick size
    # 0.00001 -> 5
    decimals = abs(tick.as_tuple().exponent)
    
    # Usar Decimal para precisión exacta
    d_price = Decimal(str(price))
    
    # Redondeo al tick size
    # El método quantize es ideal para esto
    snapped = d_price.quantize(tick, rounding=ROUND_HALF_UP)
    
    return float(snapped)

def safe_add(a: float, b: float, instrument: str = "EURUSD") -> float:
    """
    Suma dos valores y aplica snap_to_tick al resultado.
    """
    return snap_to_tick(a + b, instrument)

def pip_to_price(pips: float, instrument: str = "EURUSD") -> float:
    """
    Convierte pips a valor de precio (ej: 10 pips -> 0.00100).
    """
    # En Forex estándar (excepto JPY), 1 pip = 0.0001
    # Pero snap_to_tick garantiza la precisión del instrumento (5-digit)
    price_val = pips * 0.0001
    return snap_to_tick(price_val, instrument)

def r_to_pips(r_value: float, sl_pips: float) -> float:
    """
    Calcula pips a partir de un múltiplo R y el SL en pips.
    """
    return r_value * sl_pips

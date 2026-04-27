from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session

NAME = "ict_fvg_liquidity_gap"
WARMUP_BARS = 20
EXPLICIT_TIMEFRAME = "M15"

def parameter_space() -> dict[str, list]:
    return {
        "min_gap_pips": [0.5, 1.0, 1.5],
        "target_rr": [2.0, 3.0, 4.0],
        "displacement_atr_ratio": [1.0, 1.5, 2.0], # Criterio de "Smart Money"
        "entry_retracement": [0.0, 0.5], # 0% (limit at candle 1 high) or 50% (equilibrium)
        "session_name": ["pm_11_1630"],
    }

def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"min_gap_pips": 1.0, "target_rr": 3.0, "displacement_atr_ratio": 1.5, "entry_retracement": 0.5, "session_name": "pm_11_1630"},
        {"min_gap_pips": 1.5, "target_rr": 4.0, "displacement_atr_ratio": 2.0, "entry_retracement": 0.0, "session_name": "pm_11_1630"},
    ]

def default_params() -> dict:
    return parameter_grid(1)[0]

def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Session Filter
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # 2. Need at least 3 bars for FVG pattern
    if i < 2:
        return None

    # Structural variables (3-bar pattern)
    # i-2: Vela 1 (Extremo 1 del gap)
    # i-1: Vela 2 (Desplazamiento / Vela impulsiva)
    # i:   Vela 3 (Extremo 2 del gap)
    
    # Pre-calculate common values
    prev1_high = float(frame["high"].iat[i-2])
    prev1_low = float(frame["low"].iat[i-2])
    
    mid_high = float(frame["high"].iat[i-1])
    mid_low = float(frame["low"].iat[i-1])
    mid_close = float(frame["close"].iat[i-1])
    mid_open = float(frame["open"].iat[i-1])
    
    curr_high = float(frame["high"].iat[i])
    curr_low = float(frame["low"].iat[i])
    
    # A. displacement Check (Vela impulsiva central)
    body_size = abs(mid_close - mid_open)
    bar_range_mid = mid_high - mid_low
    atr_val = float(frame["atr14"].iat[i-1])
    
    # Debe ser una vela de "convicción": cuerpo > 1.5 * ATR
    if atr_val == 0 or (body_size / atr_val) < params.get("displacement_atr_ratio", 1.5):
        return None

    # B. Cierre en Cuartiles (Hard Spec)
    # Long: Cierre en el 25% superior de la vela
    # Short: Cierre en el 25% inferior de la vela
    if bar_range_mid == 0: return None
    relative_close = (mid_close - mid_low) / bar_range_mid

    # --- LOGIC LONG FVG (Bullish) ---
    if curr_low > prev1_high:
        gap_size = curr_low - prev1_high
        gap_pips = gap_size / 0.0001
        
        # Filtros adicionales de objetividad
        if gap_pips >= params.get("min_gap_pips", 1.0) and relative_close >= 0.75:
            # Entrada en Midpoint (Equilibrium)
            entry_price = prev1_high + (gap_size * 0.5)
            stop_price = prev1_low 
            
            return {
                "direction": "long",
                "stop_mode": "price",
                "stop_price": stop_price,
                "target_mode": "rr",
                "target_rr": params["target_rr"],
                "fvg_center": entry_price
            }

    # --- LOGIC SHORT FVG (Bearish) ---
    if curr_high < prev1_low:
        gap_size = prev1_low - curr_high
        gap_pips = gap_size / 0.0001
        
        if gap_pips >= params.get("min_gap_pips", 1.0) and relative_close <= 0.25:
            entry_price = prev1_low - (gap_size * 0.5)
            stop_price = prev1_high
            
            return {
                "direction": "short",
                "stop_mode": "price",
                "stop_price": stop_price,
                "target_mode": "rr",
                "target_rr": params["target_rr"],
                "fvg_center": entry_price
            }

    return None

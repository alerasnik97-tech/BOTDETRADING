from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session

NAME = "h1_aligned_fvg"
WARMUP_BARS = 60 # Required for H1 indicators
EXPLICIT_TIMEFRAME = "M15"

def parameter_space() -> dict[str, list]:
    return {
        "min_gap_pips": [1.0, 1.5],
        "target_rr": [3.0, 4.0],
        "displacement_atr_ratio": [1.5, 2.0],
        "h1_adx_threshold": [20, 25],
        "session_name": ["pm_11_1630"],
    }

def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"min_gap_pips": 1.0, "target_rr": 3.0, "displacement_atr_ratio": 1.5, "h1_adx_threshold": 20, "session_name": "pm_11_1630"},
        {"min_gap_pips": 1.5, "target_rr": 4.0, "displacement_atr_ratio": 2.0, "h1_adx_threshold": 25, "session_name": "pm_11_1630"},
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

    # --- HTF REGIME FILTER (H1) ---
    # Inyectado via data_loader.py (_build_h1_context)
    h1_adx = float(frame["h1_adx14"].iat[i])
    h1_close = float(frame["h1_low"].iat[i]) # Use High/Low proxy to detect if current candle is bullish/bearish
    # Actually data_loader provides h1_ema200_slope_8 which is a great trend indicator
    h1_slope = float(frame["h1_ema200_slope_8"].iat[i])
    
    # Pre-calculate common M15 values
    prev1_high = float(frame["high"].iat[i-2])
    prev1_low = float(frame["low"].iat[i-2])
    mid_close = float(frame["close"].iat[i-1])
    mid_open = float(frame["open"].iat[i-1])
    mid_high = float(frame["high"].iat[i-1])
    mid_low = float(frame["low"].iat[i-1])
    curr_low = float(frame["low"].iat[i])
    curr_high = float(frame["high"].iat[i])
    
    # A. Displacement Check (M15)
    body_size = abs(mid_close - mid_open)
    atr_val = float(frame["atr14"].iat[i-1])
    if atr_val == 0 or (body_size / atr_val) < params.get("displacement_atr_ratio", 1.5):
        return None

    # B. Cierre en Cuartiles (M15 Convicción)
    bar_range_mid = mid_high - mid_low
    if bar_range_mid == 0: return None
    relative_close = (mid_close - mid_low) / bar_range_mid

    # --- LOGIC LONG FVG ---
    # Requirements: Gap Bullish + H1 Slope Positive + H1 ADX Trend
    if curr_low > prev1_high:
        gap_size = curr_low - prev1_high
        gap_pips = gap_size / 0.0001
        
        if (gap_pips >= params.get("min_gap_pips", 1.0) and 
            relative_close >= 0.75 and 
            h1_slope > 0 and 
            h1_adx >= params.get("h1_adx_threshold", 20)):
            
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

    # --- LOGIC SHORT FVG ---
    if curr_high < prev1_low:
        gap_size = prev1_low - curr_high
        gap_pips = gap_size / 0.0001
        
        if (gap_pips >= params.get("min_gap_pips", 1.0) and 
            relative_close <= 0.25 and 
            h1_slope < 0 and 
            h1_adx >= params.get("h1_adx_threshold", 20)):
            
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

from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import add_general_params, is_in_session, stratified_sample_combinations


NAME = "midday_range_breakout_continuation"
WARMUP_BARS = 120
EXPLICIT_TIMEFRAME = "M15"


def parameter_space() -> dict[str, list]:
    return {
        "target_rr": [1.5, 2.0, 2.1],
        "break_even_at_r": [None, 1.0, 1.2],
        "session_name": ["light_fixed", "pm_11_12", "pm_12_1330", "pm_1330_16", "pm_1630_19", "pm_11_1630"],
    }


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    presets = [
        {"target_rr": 1.5, "break_even_at_r": None, "session_name": "light_fixed"},
        {"target_rr": 2.0, "break_even_at_r": 1.0, "session_name": "light_fixed"},
        {"target_rr": 2.1, "break_even_at_r": 1.2, "session_name": "light_fixed"},
    ]
    return presets


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # Filtro Horario Dinamico
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    mid_h = frame["session_range_high_11_00_13_00"].iat[i]
    mid_l = frame["session_range_low_11_00_13_00"].iat[i]
    complete = frame["session_range_complete_11_00_13_00"].iat[i]
    
    if not complete or np.isnan(mid_h) or np.isnan(mid_l):
        return None
        
    close_curr = float(frame["close"].iat[i])
    close_prev = float(frame["close"].iat[i-1])
    
    # Filtro H1 Simple
    h1_ema50 = float(frame["h1_ema50"].iat[i])
    h1_ema200 = float(frame["h1_ema200"].iat[i])
    
    # --- LOGICA LONG (Breakout Up) ---
    is_breakout_up = close_curr > mid_h and close_prev <= mid_h
    is_trend_up = h1_ema50 > h1_ema200
    
    if is_breakout_up and is_trend_up:
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": mid_l, # SL al lado opuesto del rango
            "target_mode": "rr",
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "session_name": params["session_name"],
        }
        
    # --- LOGICA SHORT (Breakout Down) ---
    is_breakout_down = close_curr < mid_l and close_prev >= mid_l
    is_trend_down = h1_ema50 < h1_ema200
    
    if is_breakout_down and is_trend_down:
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": mid_h, # SL al lado opuesto del rango
            "target_mode": "rr",
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "session_name": params["session_name"],
        }
        
    return None

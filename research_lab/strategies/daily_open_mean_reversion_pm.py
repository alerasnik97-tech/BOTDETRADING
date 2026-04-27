from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import add_general_params, is_in_session, stratified_sample_combinations


NAME = "daily_open_mean_reversion_pm"
WARMUP_BARS = 200
EXPLICIT_TIMEFRAME = "M15"


def parameter_space() -> dict[str, list]:
    return {
        "target_rr": [1.0, 1.5, 2.0],
        "break_even_at_r": [None, 1.0],
        "distance_atr_mult": [2.0, 2.5],
        "session_name": ["light_fixed", "pm_11_12", "pm_12_1330", "pm_1330_16", "pm_1630_19", "pm_11_1630"],
    }


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    presets = [
        {"target_rr": 1.5, "break_even_at_r": None, "distance_atr_mult": 2.0, "session_name": "light_fixed"},
        {"target_rr": 2.0, "break_even_at_r": 1.0, "distance_atr_mult": 2.0, "session_name": "light_fixed"},
        {"target_rr": 1.0, "break_even_at_r": None, "distance_atr_mult": 2.5, "session_name": "light_fixed"},
    ]
    return presets


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # Filtro Horario Dinamico
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    daily_open = float(frame["daily_open"].iat[i])
    atr_val = float(frame["atr14"].iat[i])
    close_curr = float(frame["close"].iat[i])
    close_prev = float(frame["close"].iat[i-1])
    
    dist = close_curr - daily_open
    abs_dist = abs(dist)
    
    threshold = params.get("distance_atr_mult", 2.0) * atr_val
    if abs_dist < threshold:
        return None
        
    # --- LOGICA SHORT (Price is high, reversión al open) ---
    if dist > 0:
        # Reingreso: vela bajista
        if close_curr < close_prev:
            return {
                "direction": "short",
                "stop_mode": "price",
                "stop_price": float(frame["high"].iat[i]) + 0.0001,
                "target_mode": "rr", # El usuario prefiere RR fijo para comparabilidad
                "target_rr": params["target_rr"],
                "break_even_at_r": params["break_even_at_r"],
                "session_name": params["session_name"],
            }
            
    # --- LOGICA LONG (Price is low, reversión al open) ---
    if dist < 0:
        # Reingreso: vela alcista
        if close_curr > close_prev:
            return {
                "direction": "long",
                "stop_mode": "price",
                "stop_price": float(frame["low"].iat[i]) - 0.0001,
                "target_mode": "rr",
                "target_rr": params["target_rr"],
                "break_even_at_r": params["break_even_at_r"],
                "session_name": params["session_name"],
            }
            
    return None

from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import add_general_params, is_in_session, stratified_sample_combinations


NAME = "adr_exhaustion_fade"
WARMUP_BARS = 200
EXPLICIT_TIMEFRAME = "M15"


def parameter_space() -> dict[str, list]:
    return {
        "target_rr": [1.5, 2.0, 2.1],
        "break_even_at_r": [None, 1.0, 1.2],
        "exhaustion_mult": [1.5, 2.0],
        "session_name": ["light_fixed", "pm_11_12", "pm_12_1330", "pm_1330_16", "pm_1630_19", "pm_11_1630"],
    }


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    presets = [
        {"target_rr": 1.5, "break_even_at_r": None, "exhaustion_mult": 1.5, "session_name": "light_fixed"},
        {"target_rr": 2.0, "break_even_at_r": 1.0, "exhaustion_mult": 1.5, "session_name": "light_fixed"},
        {"target_rr": 2.1, "break_even_at_r": 1.2, "exhaustion_mult": 1.5, "session_name": "light_fixed"},
    ]
    return presets


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # Filtro Horario Dinamico
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    atr_val = float(frame["h1_atr14"].iat[i])
    # Distancia desde el high/low del dia al extremo opuesto
    day_range = float(frame["day_running_range"].iat[i])
    
    threshold = params.get("exhaustion_mult", 1.5) * atr_val
    if day_range < threshold:
        return None
        
    close_curr = float(frame["close"].iat[i])
    high_curr = float(frame["high"].iat[i])
    low_curr = float(frame["low"].iat[i])
    
    day_high = float(frame["day_running_high"].iat[i])
    day_low = float(frame["day_running_low"].iat[i])
    
    # --- LOGICA FADE HIGH (Short) ---
    is_at_high = high_curr >= day_high
    # Reingreso objetivo: cierre por debajo del minimo de la vela anterior post-extrema
    is_reversal_short = close_curr < float(frame["low"].iat[i-1])
    
    if is_at_high and is_reversal_short:
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": high_curr + 0.0001,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "session_name": params["session_name"],
        }
        
    # --- LOGICA FADE LOW (Long) ---
    is_at_low = low_curr <= day_low
    is_reversal_long = close_curr > float(frame["high"].iat[i-1])
    
    if is_at_low and is_reversal_long:
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": low_curr - 0.0001,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "session_name": params["session_name"],
        }
        
    return None

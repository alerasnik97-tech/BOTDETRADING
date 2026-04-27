from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import add_general_params, is_in_session, stratified_sample_combinations


NAME = "asia_london_sweep_reversion_pm"
WARMUP_BARS = 120
EXPLICIT_TIMEFRAME = "M15"


def parameter_space() -> dict[str, list]:
    return {
        "target_rr": [1.5, 2.0, 2.1],
        "break_even_at_r": [None, 1.0, 1.2],
        "sweep_buffer_pips": [0.5],
        "session_name": ["light_fixed", "pm_11_12", "pm_12_1330", "pm_1330_16", "pm_1630_19", "pm_11_1630"],
    }


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    base_params = {
        "sweep_buffer_pips": 0.5,
        "session_name": "light_fixed",
    }
    presets = [
        {**base_params, "target_rr": 1.5, "break_even_at_r": None},
        {**base_params, "target_rr": 2.0, "break_even_at_r": 1.0},
        {**base_params, "target_rr": 2.1, "break_even_at_r": 1.2},
    ]
    return presets


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # Filtro Horario Dinamico
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # Niveles Asia y Londres
    asia_h = frame["session_range_high_00_00_07_00"].iat[i]
    asia_l = frame["session_range_low_00_00_07_00"].iat[i]
    lon_h = frame["session_range_high_03_00_11_00"].iat[i]
    lon_l = frame["session_range_low_03_00_11_00"].iat[i]
    
    # Combinar extremos
    all_highs = [h for h in [asia_h, lon_h] if not np.isnan(h)]
    all_lows = [l for l in [asia_l, lon_l] if not np.isnan(l)]
    
    if not all_highs or not all_lows:
        return None
        
    ext_h = max(all_highs)
    ext_l = min(all_lows)
    
    high_curr = float(frame["high"].iat[i])
    low_curr = float(frame["low"].iat[i])
    close_curr = float(frame["close"].iat[i])
    close_prev = float(frame["close"].iat[i - 1])
    close_curr = float(frame["close"].iat[i])
    high_prev = float(frame["high"].iat[i-1])
    low_prev = float(frame["low"].iat[i-1])
    
    sweep_dist = params.get("sweep_buffer_pips", 0.0) * 0.0001
    
    # Nueva Logica: Usar extremos acumulados del dia
    day_h = frame["day_running_high"].iat[i]
    day_l = frame["day_running_low"].iat[i]
    
    sweep_dist = params.get("sweep_buffer_pips", 0.0) * 0.0001
    
    # --- LOGICA SHORT (Sweep Asia/London High) ---
    # Trigger A: La vela actual barre y cierra dentro
    trigger_a = (high_curr > ext_h + sweep_dist) and (close_curr < ext_h)
    # Trigger B: La vela anterior cerro fuera y la actual cierra dentro
    trigger_b = (close_prev >= ext_h) and (close_curr < ext_h)
    
    if (trigger_a or trigger_b):
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": max(high_curr, day_h) + 0.0001,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "session_name": params["session_name"],
        }
        
    # --- LOGICA LONG (Sweep Asia/London Low) ---
    # Trigger A: La vela actual barre y cierra dentro
    trigger_a = (low_curr < ext_l - sweep_dist) and (close_curr > ext_l)
    # Trigger B: La vela anterior cerro fuera y la actual cierra dentro
    trigger_b = (close_prev <= ext_l) and (close_curr > ext_l)
    
    if (trigger_a or trigger_b):
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": min(low_curr, day_l) - 0.0001,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "session_name": params["session_name"],
        }
        
    return None

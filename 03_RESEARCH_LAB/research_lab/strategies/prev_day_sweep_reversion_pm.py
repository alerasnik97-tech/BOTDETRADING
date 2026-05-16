from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import add_general_params, is_in_session, stratified_sample_combinations


NAME = "prev_day_sweep_reversion_pm"
WARMUP_BARS = 120
EXPLICIT_TIMEFRAME = "M15"


def parameter_space() -> dict[str, list]:
    return {
        "target_rr": [1.5, 2.0, 2.1],
        "break_even_at_r": [None, 1.0, 1.2],
        "sweep_buffer_pips": [0.0, 1.0],
        "session_name": ["light_fixed", "pm_11_12", "pm_12_1330", "pm_1330_16", "pm_1630_19", "pm_11_1630"],
    }


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    # Para esta fase, queremos exactamente los 3 presets de gestion si es posible,
    # pero el sistema usa muestreo. Vamos a forzar los 3 presets principales.
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

    # 2. Niveles del dia anterior
    pdh = frame["prev_day_high"].iat[i]
    pdl = frame["prev_day_low"].iat[i]
    
    if np.isnan(pdh) or np.isnan(pdl):
        return None
        
    # 3. Datos vela actual (i) y previa (i-1)
    high_curr = float(frame["high"].iat[i])
    low_curr = float(frame["low"].iat[i])
    close_curr = float(frame["close"].iat[i])
    close_prev = float(frame["close"].iat[i-1])
    
    high_prev = float(frame["high"].iat[i-1])
    low_prev = float(frame["low"].iat[i-1])
    
    sweep_dist = params.get("sweep_buffer_pips", 0.0) * 0.0001
    
    # Nueva Logica: El barrido puede haber ocurrido en cualquier momento del dia actual
    day_h = frame["day_running_high"].iat[i]
    day_l = frame["day_running_low"].iat[i]
    
    sweep_dist = params.get("sweep_buffer_pips", 0.0) * 0.0001
    
    # --- LOGICA SHORT (Sweep PDH) ---
    # Trigger A: La vela actual barre y cierra dentro
    trigger_a = (high_curr > pdh + sweep_dist) and (close_curr < pdh)
    # Trigger B: La vela anterior cerro fuera y la actual cierra dentro
    trigger_b = (close_prev >= pdh) and (close_curr < pdh)
    
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
        
    # --- LOGICA LONG (Sweep PDL) ---
    # Trigger A: La vela actual barre y cierra dentro
    trigger_a = (low_curr < pdl - sweep_dist) and (close_curr > pdl)
    # Trigger B: La vela anterior cerro fuera y la actual cierra dentro
    trigger_b = (close_prev <= pdl) and (close_curr > pdl)
    
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

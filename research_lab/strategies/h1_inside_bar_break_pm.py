from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import add_general_params, is_in_session, stratified_sample_combinations


NAME = "h1_inside_bar_break_pm"
WARMUP_BARS = 300
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

    # H1 Context (al ser M15, cada 4 velas cambia el h1_avg_high si lo ffillamos)
    # Necesitamos detectar si la vela H1 inmediata anterior es INSIDE comparada con su previa.
    # Dado el ffill en data_loader:
    # h1_high[i] es el high de la vela H1 que cerro ANTES de i.
    
    # Para detectar el cambio de la vela H1, buscamos donde h1_high cambia.
    # Pero es mas facil: miramos el h1_high actual y el de hace 4 velas (o mas, hasta que cambie).
    
    # Buscamos el "Mother Bar" (H1 anterior) y el "Inside Bar" (H1 actual)
    # En M15, si estamos a las 11:15, h1_high[i] es la vela de las 10:00-11:00.
    # Queremos ver si esa vela (10-11) es inside de la de 09:00-10:00.
    
    # 1. Identificar los dos ultimos cierres de H1
    h1_high_curr = frame["h1_high"].iat[i]
    h1_low_curr = frame["h1_low"].iat[i]
    
    # Buscamos hacia atras el primer valor diferente y que no sea NaN
    idx = i - 1
    h1_high_prev = np.nan
    h1_low_prev = np.nan
    # Limitar la busqueda a 10 velas H1 (40 velas M15) para evitar loops infinitos si hay NaNs
    for _ in range(40):
        if idx < 0: break
        val = frame["h1_high"].iat[idx]
        if not np.isnan(val) and val != h1_high_curr:
            h1_high_prev = val
            h1_low_prev = frame["h1_low"].iat[idx]
            break
        idx -= 1
        
    if np.isnan(h1_high_prev):
        return None
        
    # 2. Verificar si es Inside Bar (La vela que acaba de cerrar es inside de la anterior)
    is_inside = h1_high_curr < h1_high_prev and h1_low_curr > h1_low_prev
    if not is_inside:
        return None
        
    # 3. Detectar Breakout en la vela M15 actual (i)
    close_curr = float(frame["close"].iat[i])
    close_prev = float(frame["close"].iat[i-1])
    
    # Breakout Up
    if close_prev <= h1_high_curr < close_curr:
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": h1_low_curr - 0.0001,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "session_name": params["session_name"],
        }
        
    # Breakout Down
    if close_prev >= h1_low_curr > close_curr:
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": h1_high_curr + 0.0001,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "session_name": params["session_name"],
        }
        
    return None

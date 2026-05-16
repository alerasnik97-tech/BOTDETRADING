from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import add_general_params, is_in_session, stratified_sample_combinations


NAME = "h1_trend_pullback_pm"
WARMUP_BARS = 200
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

    h1_ema50 = float(frame["h1_ema50"].iat[i])
    h1_ema200 = float(frame["h1_ema200"].iat[i])
    
    ema20 = float(frame["ema20"].iat[i])
    close_curr = float(frame["close"].iat[i])
    low_curr = float(frame["low"].iat[i])
    high_curr = float(frame["high"].iat[i])
    
    high_prev = float(frame["high"].iat[i-1])
    low_prev = float(frame["low"].iat[i-1])
    close_prev = float(frame["close"].iat[i-1])

    # --- LOGICA LONG ---
    is_trend_up = h1_ema50 > h1_ema200
    # Pullback: la vela actual o anterior toco/cruzo la EMA 20 por debajo
    had_pullback_up = low_curr < ema20 or low_prev < ema20
    # Confirmacion: cierre alcista que rompe el maximo de la vela anterior
    is_confirmation_up = close_curr > high_prev
    
    if is_trend_up and had_pullback_up and is_confirmation_up:
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": min(low_curr, low_prev) - 0.0001,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "session_name": params["session_name"],
        }
        
    # --- LOGICA SHORT ---
    is_trend_down = h1_ema50 < h1_ema200
    had_pullback_down = high_curr > ema20 or high_prev > ema20
    is_confirmation_down = close_curr < low_prev
    
    if is_trend_down and had_pullback_down and is_confirmation_down:
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": max(high_curr, high_prev) + 0.0001,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "session_name": params["session_name"],
        }
        
    return None

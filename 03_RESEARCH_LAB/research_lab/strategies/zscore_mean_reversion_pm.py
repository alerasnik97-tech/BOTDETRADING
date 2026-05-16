from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session


NAME = "zscore_mean_reversion_pm"
WARMUP_BARS = 100
EXPLICIT_TIMEFRAME = "M15"


def parameter_space() -> dict[str, list]:
    return {
        "z_threshold": [2.5, 3.0, 3.5],
        "lookback": [20, 40, 60],
        "target_rr": [1.0, 1.5, 2.0],
        "min_bar_atr_ratio": [0.5, 0.8, 1.0], # Calidad de vela
        "session_name": ["pm_11_1630"],
    }


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"z_threshold": 2.5, "lookback": 20, "target_rr": 1.5, "min_bar_atr_ratio": 0.8, "session_name": "pm_11_1630"},
        {"z_threshold": 3.0, "lookback": 40, "target_rr": 2.0, "min_bar_atr_ratio": 1.0, "session_name": "pm_11_1630"},
        {"z_threshold": 3.5, "lookback": 60, "target_rr": 1.5, "min_bar_atr_ratio": 0.5, "session_name": "pm_11_1630"},
    ]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Session Filter
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # 2. Calculation of Z-Score
    lookback = params["lookback"]
    if i < lookback:
        return None
        
    slice_close = frame["close"].iloc[i-lookback:i+1]
    mean = slice_close.mean()
    std = slice_close.std(ddof=0)
    
    if std == 0:
        return None
        
    z_score = (frame["close"].iat[i] - mean) / std
    threshold = params["z_threshold"]
    
    # 3. Volatility Filter (ATR Check)
    # Requerimos que la vela tenga un tamaño minimo relativo al ATR para evitar ruido
    bar_range = float(frame["high"].iat[i] - frame["low"].iat[i])
    atr_val = float(frame["atr14"].iat[i])
    if atr_val == 0 or (bar_range / atr_val) < params.get("min_bar_atr_ratio", 0.0):
        return None

    # Logic: Mean Reversion
    if z_score > threshold:
        return {
            "direction": "short",
            "stop_mode": "atr",
            "stop_atr": 2.0,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }
        
    if z_score < -threshold:
        return {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": 2.0,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }
        
    return None

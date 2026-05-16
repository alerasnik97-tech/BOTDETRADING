from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session


NAME = "adx_momentum_breakout"
WARMUP_BARS = 60
EXPLICIT_TIMEFRAME = "M15"


def parameter_space() -> dict[str, list]:
    return {
        "adx_threshold": [20, 25, 30],
        "ema_period": [20, 50],
        "target_rr": [1.5, 2.0],
        "session_name": ["pm_11_1630", "pm_11_16", "pm_11_17"], # Study windows
    }


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"adx_threshold": 25, "ema_period": 20, "target_rr": 1.5, "session_name": "pm_11_1630"},
        {"adx_threshold": 30, "ema_period": 50, "target_rr": 2.0, "session_name": "pm_11_16"},
        {"adx_threshold": 20, "ema_period": 50, "target_rr": 1.5, "session_name": "pm_11_17"},
    ]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Session Filter
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # 2. ADX Filter
    adx_val = frame["adx14"].iat[i]
    if adx_val < params["adx_threshold"]:
        return None
        
    # 3. EMA alignment
    ema_p = params["ema_period"]
    ema_val = frame[f"ema{ema_p}"].iat[i]
    close_curr = float(frame["close"].iat[i])
    open_curr = float(frame["open"].iat[i])
    
    # Logic: Momentum Breakout
    # Long: Close > EMA and Open < EMA (Breakout from below)
    if open_curr < ema_val and close_curr > ema_val:
        return {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": 2.0,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }
        
    # Short: Close < EMA and Open > EMA (Breakout from above)
    if open_curr > ema_val and close_curr < ema_val:
        return {
            "direction": "short",
            "stop_mode": "atr",
            "stop_atr": 2.0,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }
        
    return None

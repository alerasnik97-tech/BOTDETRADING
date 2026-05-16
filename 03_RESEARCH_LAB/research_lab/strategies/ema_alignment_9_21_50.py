from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session


NAME = "ema_alignment_9_21_50"
WARMUP_BARS = 100
EXPLICIT_TIMEFRAME = "M15"


def parameter_space() -> dict[str, list]:
    return {
        "target_rr": [1.5, 2.0, 3.0],
        "session_name": ["pm_11_1630", "pm_11_16", "pm_11_17"],
    }


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"target_rr": 2.0, "session_name": "pm_11_1630"},
        {"target_rr": 1.5, "session_name": "pm_11_16"},
    ]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Session Filter
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # 2. EMA Alignment (9 > 21 > 50 for Long, 9 < 21 < 50 for Short)
    ema9 = float(frame["ema9"].iat[i])
    ema21 = float(frame["ema21"].iat[i])
    ema50 = float(frame["ema50"].iat[i])
    
    # Check for crossover (we want to entry on the shift to alignment)
    if i < 1:
        return None
        
    prev_ema9 = float(frame["ema9"].iat[i-1])
    prev_ema21 = float(frame["ema21"].iat[i-1])
    
    # --- LOGIC LONG ---
    # Current Alignment + Previous bar was NOT aligned or 9 was below 21
    if ema9 > ema21 > ema50 and (prev_ema9 <= prev_ema21):
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": ema50 - 0.0001,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }
        
    # --- LOGIC SHORT ---
    # Current Alignment + Previous bar was NOT aligned or 9 was above 21
    if ema9 < ema21 < ema50 and (prev_ema9 >= prev_ema21):
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": ema50 + 0.0001,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }
        
    return None

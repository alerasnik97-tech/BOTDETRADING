from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session

NAME = "prev_day_extrema_sweep"
WARMUP_BARS = 60
EXPLICIT_TIMEFRAME = "M15"

def parameter_space() -> dict[str, list]:
    return {
        "sweep_buffer_pips": [1.0, 1.5, 2.0],
        "target_rr": [2.0, 3.0, 4.0],
        "session_name": ["pm_11_1630"],
    }

def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"sweep_buffer_pips": 1.0, "target_rr": 2.0, "session_name": "pm_11_1630"},
        {"sweep_buffer_pips": 1.5, "target_rr": 3.0, "session_name": "pm_11_1630"},
    ]

def default_params() -> dict:
    return parameter_grid(1)[0]

def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Session Filter
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # 2. Get Prev Day Levels
    pdh = float(frame["prev_day_high"].iat[i])
    pdl = float(frame["prev_day_low"].iat[i])
    
    if pd.isna(pdh) or pd.isna(pdl):
        return None

    # 3. Sweep Logic
    buffer = params.get("sweep_buffer_pips", 1.0) * 0.0001
    curr_close = float(frame["close"].iat[i])
    curr_high = float(frame["high"].iat[i])
    curr_low = float(frame["low"].iat[i])
    prev_high = float(frame["high"].iat[i-1])
    prev_low = float(frame["low"].iat[i-1])

    # SELL Sweep (PDH)
    if (curr_high > pdh + buffer or prev_high > pdh + buffer) and (curr_close < pdh):
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": max(curr_high, prev_high) + (0.5 * 0.0001),
            "target_mode": "rr",
            "target_rr": params["target_rr"]
        }

    # BUY Sweep (PDL)
    if (curr_low < pdl - buffer or prev_low < pdl - buffer) and (curr_close > pdl):
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": min(curr_low, prev_low) - (0.5 * 0.0001),
            "target_mode": "rr",
            "target_rr": params["target_rr"]
        }

    return None

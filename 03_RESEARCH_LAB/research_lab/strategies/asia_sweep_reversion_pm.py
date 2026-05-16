from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session

NAME = "asia_sweep_reversion_pm"
WARMUP_BARS = 60
EXPLICIT_TIMEFRAME = "M15"

def parameter_space() -> dict[str, list]:
    return {
        "sweep_buffer_pips": [1.0, 2.0],
        "london_climax_filter": [1.5, 2.5], # ATR multiplier for London range vs Asia range
        "target_rr": [2.0, 3.0],
        "session_name": ["pm_11_1630"],
    }

def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"sweep_buffer_pips": 1.0, "london_climax_filter": 2.5, "target_rr": 2.0, "session_name": "pm_11_1630"},
        {"sweep_buffer_pips": 2.0, "london_climax_filter": 1.5, "target_rr": 3.0, "session_name": "pm_11_1630"},
    ]

def default_params() -> dict:
    return parameter_grid(1)[0]

def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Session Filter
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # 2. Check for Asia Range (00:00 - 07:00)
    asia_high_col = "session_range_high_00_00_07_00"
    asia_low_col = "session_range_low_00_00_07_00"
    asia_complete = "session_range_complete_00_00_07_00"
    
    if asia_high_col not in frame.columns or pd.isna(frame[asia_high_col].iat[i]):
        return None
    
    if not frame[asia_complete].iat[i]:
        return None

    asia_high = float(frame[asia_high_col].iat[i])
    asia_low = float(frame[asia_low_col].iat[i])
    asia_range_size = asia_high - asia_low
    
    # 3. Climax Filter (Avoid if London expanded too much)
    # Check London Range (03:00 - 11:00)
    london_high = float(frame["session_range_high_03_00_11_00"].iat[i])
    london_low = float(frame["session_range_low_03_00_11_00"].iat[i])
    london_range_size = london_high - london_low
    
    if asia_range_size > 0:
        if (london_range_size / asia_range_size) > params.get("london_climax_filter", 2.5):
            return None

    # 4. Sweep Logic (M15 Re-entry)
    buffer = params.get("sweep_buffer_pips", 1.0) * 0.0001
    curr_close = float(frame["close"].iat[i])
    curr_high = float(frame["high"].iat[i])
    curr_low = float(frame["low"].iat[i])
    prev_high = float(frame["high"].iat[i-1])
    prev_low = float(frame["low"].iat[i-1])

    # SELL Sweep (High)
    if (curr_high > asia_high + buffer or prev_high > asia_high + buffer) and (curr_close < asia_high):
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": max(curr_high, prev_high) + (0.5 * 0.0001),
            "target_mode": "rr",
            "target_rr": params["target_rr"]
        }

    # BUY Sweep (Low)
    if (curr_low < asia_low - buffer or prev_low < asia_low - buffer) and (curr_close > asia_low):
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": min(curr_low, prev_low) - (0.5 * 0.0001),
            "target_mode": "rr",
            "target_rr": params["target_rr"]
        }

    return None

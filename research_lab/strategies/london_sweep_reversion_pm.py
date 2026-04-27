from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session

NAME = "london_sweep_reversion_pm"
WARMUP_BARS = 60
EXPLICIT_TIMEFRAME = "M15"

def parameter_space() -> dict[str, list]:
    return {
        "sweep_buffer_pips": [1.0, 1.5, 2.0],
        "target_rr": [2.0, 3.0],
        "session_name": ["pm_11_1630"],
    }

def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"sweep_buffer_pips": 1.0, "target_rr": 2.0, "session_name": "pm_11_1630"},
        {"sweep_buffer_pips": 1.0, "target_rr": 3.0, "session_name": "pm_11_1630"},
    ]

def default_params() -> dict:
    return parameter_grid(1)[0]

def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Session Filter (NY PM)
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # 2. Check for London Range Availability (03:00 - 11:00)
    high_col = "session_range_high_03_00_11_00"
    low_col = "session_range_low_03_00_11_00"
    complete_col = "session_range_complete_03_00_11_00"
    
    if high_col not in frame.columns or pd.isna(frame[high_col].iat[i]):
        return None
    
    if not frame[complete_col].iat[i]:
        return None

    london_high = float(frame[high_col].iat[i])
    london_low = float(frame[low_col].iat[i])
    buffer = params.get("sweep_buffer_pips", 1.0) * 0.0001
    
    curr_close = float(frame["close"].iat[i])
    curr_high = float(frame["high"].iat[i])
    curr_low = float(frame["low"].iat[i])
    
    # We need to track if we HAVE swept the level during the current PM session
    # but the M15 bar just closed back INSIDE.
    
    # Check for Bearish Reversion (Sweep of London High)
    # The current bar or previous PM bars must have gone above London High + Buffer
    # And the current bar must close BELOW London High.
    
    # Using a simple 1-bar lookback for the sweep trigger for objectivity
    # bar i: Close back inside
    # bar i or i-1: High was above London High + Buffer
    
    # Find PM start index
    # (Approximate since we are in at i)
    
    prev_high = float(frame["high"].iat[i-1])
    prev_low = float(frame["low"].iat[i-1])

    # --- LOGIC SELL (Sweep High) ---
    if (curr_high > london_high + buffer or prev_high > london_high + buffer) and (curr_close < london_high):
        # Additional safety: Make sure we aren't sweeping Low at the same time (choppy)
        if curr_low > london_low:
            return {
                "direction": "short",
                "stop_mode": "price",
                "stop_price": max(curr_high, prev_high) + (0.5 * 0.0001), # 0.5 pip slack
                "target_mode": "rr",
                "target_rr": params["target_rr"]
            }

    # --- LOGIC BUY (Sweep Low) ---
    if (curr_low < london_low - buffer or prev_low < london_low - buffer) and (curr_close > london_low):
        if curr_high < london_high:
            return {
                "direction": "long",
                "stop_mode": "price",
                "stop_price": min(curr_low, prev_low) - (0.5 * 0.0001),
                "target_mode": "rr",
                "target_rr": params["target_rr"]
            }

    return None

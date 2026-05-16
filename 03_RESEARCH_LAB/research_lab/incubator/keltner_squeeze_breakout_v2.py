from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session

NAME = "keltner_squeeze_breakout_v2"
WARMUP_BARS = 100
EXPLICIT_TIMEFRAME = "M15"

def parameter_space() -> dict[str, list]:
    return {
        "bb_period": [20],
        "bb_std": [2.0],
        "kc_period": [20],
        "kc_mult": [1.5],
        "target_rr": [1.5, 2.0, 3.0],
        "session_name": ["pm_11_1630", "pm_11_16", "pm_11_17"],
    }

def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"bb_period": 20, "bb_std": 2.0, "kc_period": 20, "kc_mult": 1.5, "target_rr": 2.0, "session_name": "pm_11_1630"},
    ]

def default_params() -> dict:
    return parameter_grid(1)[0]

def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Session Filter
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # 2. Squeeze Condition (BB inside KC)
    # suffix formats are: {period}_{std_mult} with '.' replaced by '_'
    bb_suffix = "20_2_0"
    kc_suffix = "20_1_5"
    
    bb_upper = float(frame[f"bb_upper_{bb_suffix}"].iat[i])
    bb_lower = float(frame[f"bb_lower_{bb_suffix}"].iat[i])
    kc_upper = float(frame[f"kc_upper_{kc_suffix}"].iat[i])
    kc_lower = float(frame[f"kc_lower_{kc_suffix}"].iat[i])
    
    # Check if we ARE in a squeeze
    is_squeeze = (bb_upper < kc_upper) and (bb_lower > kc_lower)
    
    # We look for a BREAKOUT from a squeeze
    # Need previous state
    prev_bb_upper = float(frame[f"bb_upper_{bb_suffix}"].iat[i-1])
    prev_bb_lower = float(frame[f"bb_lower_{bb_suffix}"].iat[i-1])
    prev_kc_upper = float(frame[f"kc_upper_{kc_suffix}"].iat[i-1])
    prev_kc_lower = float(frame[f"kc_lower_{kc_suffix}"].iat[i-1])
    was_squeeze = (prev_bb_upper < prev_kc_upper) and (prev_bb_lower > prev_kc_lower)
    
    curr_close = float(frame["close"].iat[i])
    
    # --- LOGIC LONG (Bullish Breakout) ---
    if was_squeeze and curr_close > kc_upper:
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": kc_lower, # Use the opposite channel as stop
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }

    # --- LOGIC SHORT (Bearish Breakout) ---
    if was_squeeze and curr_close < kc_lower:
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": kc_upper, # Use the opposite channel as stop
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }

    return None

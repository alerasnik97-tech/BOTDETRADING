from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session


NAME = "nr7_breakout"
WARMUP_BARS = 200
EXPLICIT_TIMEFRAME = "M15"


def parameter_space() -> dict[str, list]:
    return {
        "ema_filter": [50, 100, 200],
        "target_rr": [1.5, 2.0],
        "session_name": ["pm_11_1630", "pm_11_16", "pm_11_17"],
    }


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"ema_filter": 200, "target_rr": 2.0, "session_name": "pm_11_1630"},
        {"ema_filter": 100, "target_rr": 1.5, "session_name": "pm_11_16"},
    ]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Session Filter
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # 2. NR7 Trigger (Check PREVIOUS bar was NR7)
    if i < 1 or not frame["is_nr7"].iat[i-1]:
        return None
        
    # 3. EMA Trend Filter
    ema_period = params["ema_filter"]
    ema_col = f"ema{ema_period}"
    if ema_col not in frame.columns:
        return None
        
    curr_close = float(frame["close"].iat[i])
    ema_val = float(frame[ema_col].iat[i])
    
    # Range of the NR7 bar
    nr7_high = float(frame["high"].iat[i-1])
    nr7_low = float(frame["low"].iat[i-1])
    
    # --- LOGIC LONG ---
    # Breakout of NR7 High + Above EMA
    if curr_close > nr7_high and curr_close > ema_val:
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": nr7_low - 0.0001,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }
        
    # --- LOGIC SHORT ---
    # Breakout of NR7 Low + Below EMA
    if curr_close < nr7_low and curr_close < ema_val:
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": nr7_high + 0.0001,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }
        
    return None

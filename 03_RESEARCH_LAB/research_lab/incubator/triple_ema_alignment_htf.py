from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session

NAME = "triple_ema_alignment_htf"
WARMUP_BARS = 300
EXPLICIT_TIMEFRAME = "M15"

def parameter_space() -> dict[str, list]:
    return {
        "fast_period": [9, 21],
        "target_rr": [1.5, 2.0, 3.0],
        "session_name": ["pm_11_1630", "pm_11_16", "pm_11_17"],
    }

def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"fast_period": 9, "target_rr": 2.0, "session_name": "pm_11_1630"},
    ]

def default_params() -> dict:
    return parameter_grid(1)[0]

def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Session Filter
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # 2. HTF Trend Filters (from H1 context in data_loader)
    h1_ema50 = float(frame["h1_ema50"].iat[i])
    h1_ema100 = float(frame["h1_ema100"].iat[i])
    h1_ema200 = float(frame["h1_ema200"].iat[i])
    
    # Check for HTF Alignment
    bullish_htf = h1_ema50 > h1_ema100 > h1_ema200
    bearish_htf = h1_ema50 < h1_ema100 < h1_ema200
    
    # 3. Execution Signal (M15 Pullback)
    curr_close = float(frame["close"].iat[i])
    fast_ema = float(frame[f"ema{params['fast_period']}"].iat[i])
    
    # --- LOGIC LONG (HTF Trend + Pullback) ---
    if bullish_htf and curr_close > fast_ema:
        # Check if it was BELOW the ema before (pullback/crossover)
        prev_close = float(frame["close"].iat[i-1])
        if prev_close <= float(frame[f"ema{params['fast_period']}"].iat[i-1]):
            return {
                "direction": "long",
                "stop_mode": "atr",
                "stop_atr": 1.5,
                "target_mode": "rr",
                "target_rr": params["target_rr"],
            }

    # --- LOGIC SHORT (HTF Trend + Pullback) ---
    if bearish_htf and curr_close < fast_ema:
        prev_close = float(frame["close"].iat[i-1])
        if prev_close >= float(frame[f"ema{params['fast_period']}"].iat[i-1]):
            return {
                "direction": "short",
                "stop_mode": "atr",
                "stop_atr": 1.5,
                "target_mode": "rr",
                "target_rr": params["target_rr"],
            }

    return None

from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session


NAME = "turtle_soup_fade"
WARMUP_BARS = 60
EXPLICIT_TIMEFRAME = "M15"


def parameter_space() -> dict[str, list]:
    return {
        "lookback": [20, 30, 40],
        "target_rr": [1.5, 2.0, 3.0],
        "session_name": ["pm_11_1630", "pm_11_16", "pm_11_17"],
    }


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"lookback": 20, "target_rr": 2.0, "session_name": "pm_11_1630"},
        {"lookback": 30, "target_rr": 2.1, "session_name": "pm_11_1630"},
        {"lookback": 20, "target_rr": 1.5, "session_name": "pm_11_16"},
    ]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Session Filter
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # 2. Donchian Levels (Already in data_loader)
    lookback = params["lookback"]
    # We need the PREVIOUS donchian levels (the ones that were active before this bar)
    prev_high = frame[f"donchian_high_{lookback}"].iat[i]
    prev_low = frame[f"donchian_low_{lookback}"].iat[i]
    
    if np.isnan(prev_high) or np.isnan(prev_low):
        return None
        
    # 3. Current Price Action
    curr_high = float(frame["high"].iat[i])
    curr_low = float(frame["low"].iat[i])
    curr_close = float(frame["close"].iat[i])
    
    # --- LOGIC LONG (Turtle Soup at Low) ---
    # Price must have traded below prev_low during this bar
    # And closed back above prev_low
    if curr_low < prev_low and curr_close > prev_low:
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": curr_low - 0.0001,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }
        
    # --- LOGIC SHORT (Turtle Soup at High) ---
    # Price must have traded above prev_high during this bar
    # And closed back below prev_high
    if curr_high > prev_high and curr_close < prev_high:
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": curr_high + 0.0001,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }
        
    return None

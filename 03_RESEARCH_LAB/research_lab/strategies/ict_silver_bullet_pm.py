from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session


NAME = "ict_silver_bullet_pm"
WARMUP_BARS = 100
EXPLICIT_TIMEFRAME = "M15"


def parameter_space() -> dict[str, list]:
    return {
        "target_rr": [1.5, 2.0, 3.0],
        "break_even_at_r": [None, 1.0, 1.5],
        "session_name": ["pm_silver_bullet"], # 14:00 - 15:00 NY
    }


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"target_rr": 2.0, "break_even_at_r": 1.0, "session_name": "pm_silver_bullet"},
        {"target_rr": 3.0, "break_even_at_r": 1.5, "session_name": "pm_silver_bullet"},
        {"target_rr": 1.5, "break_even_at_r": None, "session_name": "pm_silver_bullet"},
    ]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. ICT Silver Bullet Window (14:00 - 15:00 NY)
    ts = frame.index[i]
    if not (ts.hour == 14): # M15 candles 14:00, 14:15, 14:30, 14:45
        return None

    # 2. Liquidity Reference: Lunch Range (11:00 - 13:00)
    # Provided by data_loader: session_range_high_11_00_13_00
    lunch_high = frame["session_range_high_11_00_13_00"].iat[i]
    lunch_low = frame["session_range_low_11_00_13_00"].iat[i]
    
    if np.isnan(lunch_high) or np.isnan(lunch_low):
        return None
        
    # 3. Current price action
    high_curr = float(frame["high"].iat[i])
    low_curr = float(frame["low"].iat[i])
    close_curr = float(frame["close"].iat[i])
    
    # Logic: Sweep and Reversal
    # Short: Candle swept lunch_high and closed below it
    if high_curr > lunch_high and close_curr < lunch_high:
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": high_curr + 0.0001,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
        }
        
    # Long: Candle swept lunch_low and closed above it
    if low_curr < lunch_low and close_curr > lunch_low:
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": low_curr - 0.0001,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
        }
        
    return None

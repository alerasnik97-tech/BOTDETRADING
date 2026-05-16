from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session


NAME = "donchian_intraday_breakout"
WARMUP_BARS = 60
EXPLICIT_TIMEFRAME = "M15"


def parameter_space() -> dict[str, list]:
    return {
        "bars": [20, 30, 55],
        "target_rr": [1.5, 2.0, 3.0],
        "use_ema_filter": [True, False],
        "session_name": ["pm_11_1630", "pm_11_16", "pm_11_17"], # Study windows
    }


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"bars": 20, "target_rr": 2.0, "use_ema_filter": True, "session_name": "pm_11_1630"},
        {"bars": 30, "target_rr": 2.0, "use_ema_filter": True, "session_name": "pm_11_16"},
        {"bars": 55, "target_rr": 3.0, "use_ema_filter": False, "session_name": "pm_11_17"},
    ]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Session Filter
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # 2. Donchian Levels
    bars = params["bars"]
    high_level = frame[f"donchian_high_{bars}"].iat[i]
    low_level = frame[f"donchian_low_{bars}"].iat[i]
    
    if np.isnan(high_level) or np.isnan(low_level):
        return None
        
    # 3. EMA Filter
    if params["use_ema_filter"]:
        ema200 = frame["ema200"].iat[i]
        price = frame["close"].iat[i]
        trend_up = price > ema200
        trend_down = price < ema200
    else:
        trend_up = True
        trend_down = True
        
    # 4. Current price
    high_curr = float(frame["high"].iat[i])
    low_curr = float(frame["low"].iat[i])
    close_curr = float(frame["close"].iat[i])
    
    # Logic: Breakout
    # Long: Break high level
    if trend_up and high_curr > high_level and close_curr > high_level:
        return {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": 1.5,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }
        
    # Short: Break low level
    if trend_down and low_curr < low_level and close_curr < low_level:
        return {
            "direction": "short",
            "stop_mode": "atr",
            "stop_atr": 1.5,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }
        
    return None

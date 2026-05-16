from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session

NAME = "vwap_mean_reversion_adaptive"
WARMUP_BARS = 50
EXPLICIT_TIMEFRAME = "M15"

def parameter_space() -> dict[str, list]:
    return {
        "zscore_threshold": [2.0, 2.5, 3.0],
        "target_rr": [1.0, 1.5, 2.0],
        "session_name": ["pm_11_1630", "pm_11_16", "pm_11_17"],
    }

def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"zscore_threshold": 2.5, "target_rr": 1.5, "session_name": "pm_11_1630"},
        {"zscore_threshold": 3.0, "target_rr": 1.0, "session_name": "pm_11_16"},
    ]

def default_params() -> dict:
    return parameter_grid(1)[0]

def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Session Filter
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # 2. VWAP distance (Z-Score)
    # The data_loader provides 'vwap_dist_std' which is (price - vwap) / std
    if "vwap_dist_std" not in frame.columns:
        return None
        
    zscore = float(frame["vwap_dist_std"].iat[i])
    threshold = params["zscore_threshold"]
    vwap_val = float(frame["vwap"].iat[i])
    curr_close = float(frame["close"].iat[i])
    
    # --- LOGIC LONG (Mean Reversion at Lower Band) ---
    if zscore < -threshold:
        return {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": 2.0,
            "target_mode": "price",
            "target_price": vwap_val, # Target is the mean (VWAP)
        }

    # --- LOGIC SHORT (Mean Reversion at Upper Band) ---
    if zscore > threshold:
        return {
            "direction": "short",
            "stop_mode": "atr",
            "stop_atr": 2.0,
            "target_mode": "price",
            "target_price": vwap_val, # Target is the mean (VWAP)
        }

    return None

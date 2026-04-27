from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session


NAME = "triple_macd_filter"
WARMUP_BARS = 100
EXPLICIT_TIMEFRAME = "M15"


def parameter_space() -> dict[str, list]:
    return {
        "alignment_threshold": [0.0],
        "target_rr": [1.5, 2.0, 3.0],
        "session_name": ["pm_11_1630", "pm_11_16", "pm_11_17"],
    }


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"alignment_threshold": 0.0, "target_rr": 2.0, "session_name": "pm_11_1630"},
        {"alignment_threshold": 0.0, "target_rr": 1.5, "session_name": "pm_11_16"},
    ]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Session Filter
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # 2. MACD Histograms (Calculated in data_loader)
    h_fast = float(frame["macd_fast_hist"].iat[i])
    h_main = float(frame["macd_main_hist"].iat[i])
    h_slow = float(frame["macd_slow_hist"].iat[i])
    
    # Check for previous values for crossover/momentum
    if i < 1:
        return None
        
    prev_fast = float(frame["macd_fast_hist"].iat[i-1])
    
    # --- LOGIC LONG ---
    # All histograms above zero + fast one is increasing (momentum)
    if h_fast > 0 and h_main > 0 and h_slow > 0 and h_fast > prev_fast:
        return {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": 2.0,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }
        
    # --- LOGIC SHORT ---
    # All histograms below zero + fast one is decreasing (momentum)
    if h_fast < 0 and h_main < 0 and h_slow < 0 and h_fast < prev_fast:
        return {
            "direction": "short",
            "stop_mode": "atr",
            "stop_atr": 2.0,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }
        
    return None

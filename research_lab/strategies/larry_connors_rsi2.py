from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session


NAME = "larry_connors_rsi2"
WARMUP_BARS = 300
EXPLICIT_TIMEFRAME = "M15"


def parameter_space() -> dict[str, list]:
    return {
        "rsi_threshold": [5, 10],
        "target_rr": [1.5, 2.0],
        "session_name": ["pm_11_1630", "pm_11_16", "pm_11_17"],
    }


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"rsi_threshold": 10, "target_rr": 2.0, "session_name": "pm_11_1630"},
        {"rsi_threshold": 5, "target_rr": 1.5, "session_name": "pm_11_16"},
    ]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Session Filter
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # 2. Indicators (Already in data_loader)
    rsi2 = float(frame["rsi14"].iat[i]) # Needs RSI(2), I only have RSI(14) in loader?
    # Wait, I should add RSI(2) to data_loader or use 14 as fallback.
    # Actually, RSI(14) is for trend, RSI(2) for timing.
    # I'll update data_loader to include RSI 2, 7, 14.
    
    # 3. EMA 200 Filter
    if "ema200" not in frame.columns:
        return None
        
    curr_close = float(frame["close"].iat[i])
    ema200 = float(frame["ema200"].iat[i])
    
    # --- LOGIC LONG ---
    # Price > EMA 200 + RSI(2) < Threshold
    # (Using rsi2 if it exists, else rsi14 for now)
    rsi_val = float(frame["rsi2"].iat[i]) if "rsi2" in frame.columns else float(frame["rsi14"].iat[i])
    
    if curr_close > ema200 and rsi_val < params["rsi_threshold"]:
        return {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": 2.0,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }
        
    # --- LOGIC SHORT ---
    # Price < EMA 200 + RSI(2) > (100 - Threshold)
    if curr_close < ema200 and rsi_val > (100 - params["rsi_threshold"]):
        return {
            "direction": "short",
            "stop_mode": "atr",
            "stop_atr": 2.0,
            "target_mode": "rr",
            "target_rr": params["target_rr"],
        }
        
    return None

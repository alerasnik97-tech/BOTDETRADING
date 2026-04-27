from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session

NAME = "h1_trend_pullback_v2"
WARMUP_BARS = 200 # Required for EMA 200
EXPLICIT_TIMEFRAME = "M15"

def parameter_space() -> dict[str, list]:
    return {
        "target_rr": [2.0, 3.0],
        "vwap_filter": [True, False],
        "session_name": ["pm_11_1630"],
    }

def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"target_rr": 2.0, "vwap_filter": True, "session_name": "pm_11_1630"},
        {"target_rr": 3.0, "vwap_filter": True, "session_name": "pm_11_1630"},
    ]

def default_params() -> dict:
    return parameter_grid(1)[0]

def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Session Filter
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # 2. HTF Trend Filter (H1)
    # Inyectado via data_loader.py
    h1_ema50 = float(frame["h1_ema50"].iat[i])
    h1_ema100 = float(frame["h1_ema100"].iat[i])
    h1_ema200 = float(frame["h1_ema200"].iat[i])
    h1_slope = float(frame["h1_ema200_slope_8"].iat[i])
    
    is_h1_bullish = h1_ema50 > h1_ema100 > h1_ema200 and h1_slope > 0
    is_h1_bearish = h1_ema50 < h1_ema100 < h1_ema200 and h1_slope < 0
    
    if not is_h1_bullish and not is_h1_bearish:
        return None

    # 3. Intraday Pullback Setup (M15)
    # EMA 9 crosses EMA 21
    ema9_curr = float(frame["ema9"].iat[i])
    ema21_curr = float(frame["ema21"].iat[i])
    ema9_prev = float(frame["ema9"].iat[i-1])
    ema21_prev = float(frame["ema21"].iat[i-1])
    
    vwap = float(frame["vwap"].iat[i])
    close = float(frame["close"].iat[i])
    atr = float(frame["atr14"].iat[i])
    
    # Check for Long Pullback
    if is_h1_bullish and ema9_prev <= ema21_prev and ema9_curr > ema21_curr:
        # VWAP Discount Filter: Price should be near or below VWAP to avoid buying tops
        if params["vwap_filter"] and close > vwap:
            return None
            
        return {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": 2.0,
            "target_mode": "rr",
            "target_rr": params["target_rr"]
        }

    # Check for Short Pullback
    if is_h1_bearish and ema9_prev >= ema21_prev and ema9_curr < ema21_curr:
        # VWAP Premium Filter: Price should be near or above VWAP
        if params["vwap_filter"] and close < vwap:
            return None
            
        return {
            "direction": "short",
            "stop_mode": "atr",
            "stop_atr": 2.0,
            "target_mode": "rr",
            "target_rr": params["target_rr"]
        }

    return None

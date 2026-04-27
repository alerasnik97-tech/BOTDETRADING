from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session

NAME = "h1_gated_zscore"
WARMUP_BARS = 60
EXPLICIT_TIMEFRAME = "M15"

def parameter_space() -> dict[str, list]:
    return {
        "z_threshold": [2.5, 3.0],
        "lookback": [20, 40],
        "h1_adx_limit": [25, 30],
        "session_name": ["pm_11_1630"],
    }

def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return [
        {"z_threshold": 2.5, "lookback": 20, "h1_adx_limit": 25, "session_name": "pm_11_1630"},
        {"z_threshold": 3.0, "lookback": 40, "h1_adx_limit": 30, "session_name": "pm_11_1630"},
    ]

def default_params() -> dict:
    return parameter_grid(1)[0]

def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Session Filter
    if not is_in_session(frame.index[i], params["session_name"]):
        return None

    # 2. HTF Regime Inhibitor (H1)
    # If H1 is in a strong trend (ADX > limit), we ignore mean reversion signals
    h1_adx = float(frame["h1_adx14"].iat[i])
    if h1_adx > params.get("h1_adx_limit", 25):
        return None

    # 3. Setup M15: Z-Score
    lookback = params["lookback"]
    if i < lookback: return None
    
    series = frame["close"].iloc[i-lookback:i]
    mean = series.mean()
    std = series.std()
    
    if std == 0: return None
    z_score = (frame["close"].iat[i] - mean) / std
    
    # 4. Entry Logic with Convergence
    # We only sell if we are also at the upper part of the H1 context or vice versa
    # (Optional: check if price is far from h1_ema50)
    
    if z_score > params["z_threshold"]:
        return {
            "direction": "short",
            "stop_mode": "atr",
            "stop_atr": 2.0,
            "target_mode": "rr",
            "target_rr": 1.5
        }
    
    if z_score < -params["z_threshold"]:
        return {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": 2.0,
            "target_mode": "rr",
            "target_rr": 1.5
        }

    return None

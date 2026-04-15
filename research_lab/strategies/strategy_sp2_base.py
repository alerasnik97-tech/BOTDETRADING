from __future__ import annotations
import pandas as pd
from typing import Any
from research_lab.strategies.common import add_general_params, stratified_sample_combinations

NAME = "sp2_base"
WARMUP_BARS = 60
EXPLICIT_TIMEFRAME = "M5"

def parameter_space() -> dict[str, list]:
    return add_general_params({
        "lookback_bars": [20],
        "limit_expiry_bars": [12],
        "stop_atr": [1.5],
        "target_rr": [1.5, 2.0],
        "body_pct_filter": [0.5],
        "ema_period": [50],
    })

def parameter_grid(max_combinations: int = 2, seed: int = 42) -> list[dict]:
    params_dict = parameter_space()
    params_dict["session_name"] = ["light_fixed"] # 11:00 to 15:00
    return stratified_sample_combinations(params_dict, max_combinations, seed)

def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    curr_hour = frame.index[i].hour
    if curr_hour < 11 or curr_hour >= 18:
        return None

    lookback = params["lookback_bars"]
    expiry = params["limit_expiry_bars"]
    body_pct = params["body_pct_filter"]
    ema_period = params["ema_period"]
    ema_col = f"ema{ema_period}"
    
    for j in range(i - expiry, i):
        if j < lookback: continue
        high_j, low_j = float(frame["high"].iat[j]), float(frame["low"].iat[j])
        close_j, open_j = float(frame["close"].iat[j]), float(frame["open"].iat[j])
        ema_val_j = float(frame[ema_col].iat[j])
        
        range_j = high_j - low_j
        if range_j < 0.0001: continue
        body_j = abs(close_j - open_j)
        if body_j / range_j < body_pct: continue
        
        idx_start = j - lookback
        res_R = float(frame["high"].iloc[idx_start:j].max())
        sup_S = float(frame["low"].iloc[idx_start:j].min())
        
        # Breakout LONG respect to M5 EMA
        if close_j > ema_val_j and float(frame["close"].iat[j-1]) <= res_R and close_j > res_R:
            if i > j + 1 and float(frame["low"].iloc[j+1:i].min()) < res_R - 0.0002: continue
            if float(frame["low"].iat[i]) <= res_R:
                return {
                    "direction": "long", "stop_mode": "atr", "stop_atr": params["stop_atr"],
                    "target_rr": params["target_rr"], "max_hold_bars": 12,
                    "session_name": params["session_name"], "break_even_at_r": 0.0, "signal_price": res_R
                }
                
        # Breakout SHORT respect to M5 EMA
        if close_j < ema_val_j and float(frame["close"].iat[j-1]) >= sup_S and close_j < sup_S:
            if i > j + 1 and float(frame["high"].iloc[j+1:i].max()) > sup_S + 0.0002: continue
            if float(frame["high"].iat[i]) >= sup_S:
                return {
                    "direction": "short", "stop_mode": "atr", "stop_atr": params["stop_atr"],
                    "target_rr": params["target_rr"], "max_hold_bars": 12,
                    "session_name": params["session_name"], "break_even_at_r": 0.0, "signal_price": sup_S
                }
    return None

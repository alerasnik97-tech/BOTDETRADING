from __future__ import annotations

import pandas as pd
from typing import Any
from research_lab.strategies.common import add_general_params, stratified_sample_combinations

NAME = "ny_br_pure"
WARMUP_BARS = 50
EXPLICIT_TIMEFRAME = "M5"

def parameter_space() -> dict[str, list]:
    return add_general_params(
        {
            "lookback_bars": [20],
            "limit_expiry_bars": [12], # ~1 hora para retest
            "stop_atr": [1.5],
            "target_rr": [1.5, 2.0],
            "body_pct_filter": [0.5],
        }
    )

def parameter_grid(max_combinations: int = 2, seed: int = 42) -> list[dict]:
    params_dict = parameter_space()
    params_dict["session_name"] = ["light_fixed"] # 11:00 a 19:00 implícito
    return stratified_sample_combinations(params_dict, max_combinations, seed)

def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    curr_hour = frame.index[i].hour
    if curr_hour < 11 or curr_hour >= 18:
        return None

    lookback = params["lookback_bars"]
    expiry = params["limit_expiry_bars"]
    body_pct = params["body_pct_filter"]

    # Buscar si en las ultimas 'expiry' velas hubo un breakout claro
    # Restringido para que no demore infinito en loops
    for j in range(i - expiry, i):
        if j < lookback:
            continue
            
        high_j = float(frame["high"].iat[j])
        low_j = float(frame["low"].iat[j])
        close_j = float(frame["close"].iat[j])
        open_j = float(frame["open"].iat[j])
        
        range_j = high_j - low_j
        if range_j < 0.0001: continue
        
        body_j = abs(close_j - open_j)
        if body_j / range_j < body_pct: continue # Velas decisivas solamente
        
        # Resistencia y Soporte previo (excluyendo j)
        idx_start = j - lookback
        res_R = float(frame["high"].iloc[idx_start:j].max())
        sup_S = float(frame["low"].iloc[idx_start:j].min())
        
        # Breakout LONG
        if float(frame["close"].iat[j-1]) <= res_R and close_j > res_R:
            # Check invalidation from j+1 to i-1
            if i > j + 1:
                min_low_since = float(frame["low"].iloc[j+1:i].min())
                if min_low_since < res_R - 0.0002: # Tolerancia de slip
                    continue # Ya violó demasiado abajo
                    
            # Gatillo exacto en i? (El precio baja a tocar la linea R)
            low_i = float(frame["low"].iat[i])
            if low_i <= res_R:
                return {
                    "direction": "long",
                    "stop_mode": "atr",
                    "stop_atr": params["stop_atr"],
                    "target_rr": params["target_rr"],
                    "max_hold_bars": 12,
                    "session_name": params["session_name"],
                    "break_even_at_r": 0.0,
                    "signal_price": res_R
                }
                
        # Breakout SHORT
        if float(frame["close"].iat[j-1]) >= sup_S and close_j < sup_S:
            if i > j + 1:
                max_high_since = float(frame["high"].iloc[j+1:i].max())
                if max_high_since > sup_S + 0.0002:
                    continue
                    
            high_i = float(frame["high"].iat[i])
            if high_i >= sup_S:
                return {
                    "direction": "short",
                    "stop_mode": "atr",
                    "stop_atr": params["stop_atr"],
                    "target_rr": params["target_rr"],
                    "max_hold_bars": 12,
                    "session_name": params["session_name"],
                    "break_even_at_r": 0.0,
                    "signal_price": sup_S
                }

    return None

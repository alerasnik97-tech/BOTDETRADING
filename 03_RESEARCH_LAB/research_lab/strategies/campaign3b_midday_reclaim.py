"""
Campaign 3B - M1: Midday Reclaim
Ventana: 11:00 - 13:00 NY
Timeframe: M15
Lógica: Reclaim de nivel diario/prev_day con confirmación objetiva
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session

NAME = "campaign3b_midday_reclaim"
WARMUP_BARS = 50
EXPLICIT_TIMEFRAME = "M15"
PIP_SIZE = 0.0001

DEFAULT_GRID = [
    {
        "session_name": "pm_11_17",
        "entry_start": "11:00",
        "entry_end": "13:00",
        "target_rr": 2.0,
        "break_even_at_r": 1.0,
        "max_hold_bars": 12,
        "cooldown_bars": 6,
        "atr_mult_sl": 1.5,
        "confirm_bars": 1,
    }
]

def parameter_space() -> dict[str, list]:
    return {
        "atr_mult_sl": [1.3, 1.5, 1.8],
        "confirm_bars": [1, 2],
        "target_rr": [1.8, 2.0, 2.2],
    }

def parameter_grid(max_combinations: int = 1, seed: int = 42) -> list[dict]:
    import itertools
    import random
    random.seed(seed)
    
    space = parameter_space()
    keys = list(space.keys())
    values = list(space.values())
    
    all_combos = list(itertools.product(*values))
    random.shuffle(all_combos)
    
    grid = []
    for combo in all_combos[:max_combinations]:
        params = dict(zip(keys, combo))
        grid.append({**DEFAULT_GRID[0], **params})
    
    return grid if grid else DEFAULT_GRID

def default_params() -> dict:
    return DEFAULT_GRID[0].copy()

def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    if i < WARMUP_BARS:
        return None
    
    session_name = params.get("session_name", "pm_11_17")
    atr_mult = params.get("atr_mult_sl", 1.5)
    confirm_bars = params.get("confirm_bars", 1)
    target_rr = params.get("target_rr", 2.0)
    break_even_at_r = params.get("break_even_at_r", 1.0)
    
    # Verificar sesión midday
    if not is_in_session(frame.index[i], session_name):
        return None
    
    # Niveles diarios
    prev_high = frame["prev_day_high"].iloc[i]
    prev_low = frame["prev_day_low"].iloc[i]
    atr = frame["atr14"].iloc[i]
    
    if pd.isna(prev_high) or pd.isna(prev_low) or pd.isna(atr) or not np.isfinite(atr):
        return None
    
    current_close = frame["close"].iloc[i]
    current_high = frame["high"].iloc[i]
    current_low = frame["low"].iloc[i]
    
    if pd.isna(current_close) or pd.isna(current_high) or pd.isna(current_low):
        return None
    if not np.isfinite(current_close) or not np.isfinite(current_high) or not np.isfinite(current_low):
        return None
    
    # Reclaim de prev_day_high (long)
    if i >= confirm_bars:
        # Buscar reclaim: precio rompió prev_high y volvió
        prev_bars_high = frame["high"].iloc[i-confirm_bars:i]
        prev_bars_close = frame["close"].iloc[i-confirm_bars:i]
        
        # Si hubo break arriba de prev_high
        if prev_bars_high.max() > prev_high:
            # Y ahora el precio está por debajo de prev_high (reclaim)
            if current_close < prev_high and current_low < prev_high:
                # Confirmación: cierre por debajo del nivel
                return {
                    "direction": "short",
                    "stop_atr": atr_mult,
                    "entry_price": current_close,
                    "stop_mode": "atr",
                    "target_rr": target_rr,
                    "break_even_at_r": break_even_at_r,
                }
    
    # Reclaim de prev_day_low (short)
    if i >= confirm_bars:
        prev_bars_low = frame["low"].iloc[i-confirm_bars:i]
        prev_bars_close = frame["close"].iloc[i-confirm_bars:i]
        
        # Si hubo break abajo de prev_low
        if prev_bars_low.min() < prev_low:
            # Y ahora el precio está por encima de prev_low (reclaim)
            if current_close > prev_low and current_high > prev_low:
                # Confirmación: cierre por encima del nivel
                return {
                    "direction": "long",
                    "stop_atr": atr_mult,
                    "entry_price": current_close,
                    "stop_mode": "atr",
                    "target_rr": target_rr,
                    "break_even_at_r": break_even_at_r,
                }
    
    return None

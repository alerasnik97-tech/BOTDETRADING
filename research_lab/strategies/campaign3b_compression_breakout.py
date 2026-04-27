"""
Campaign 3B - M2: Compression Breakout
Ventana: 13:00 - 17:00 NY
Timeframe: M15
Lógica: Compresión medible + expansión real
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session

NAME = "campaign3b_compression_breakout"
WARMUP_BARS = 50
EXPLICIT_TIMEFRAME = "M15"
PIP_SIZE = 0.0001

DEFAULT_GRID = [
    {
        "session_name": "pm_11_17",
        "entry_start": "13:00",
        "entry_end": "17:00",
        "target_rr": 2.5,
        "break_even_at_r": 1.0,
        "max_hold_bars": 16,
        "cooldown_bars": 8,
        "atr_mult_sl": 2.0,
        "compression_threshold_atr": 0.5,
        "compression_bars": 3,
    }
]

def parameter_space() -> dict[str, list]:
    return {
        "atr_mult_sl": [1.8, 2.0, 2.2],
        "compression_threshold_atr": [0.4, 0.5, 0.6],
        "compression_bars": [2, 3, 4],
        "target_rr": [2.2, 2.5, 2.8],
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
    atr_mult = params.get("atr_mult_sl", 2.0)
    compression_threshold = params.get("compression_threshold_atr", 0.5)
    compression_bars = params.get("compression_bars", 3)
    target_rr = params.get("target_rr", 2.5)
    break_even_at_r = params.get("break_even_at_r", 1.0)
    
    # Verificar sesión afternoon
    if not is_in_session(frame.index[i], session_name):
        return None
    
    # Verificar compresión
    if i < compression_bars + 1:
        return None
    
    atr = frame["atr14"].iloc[i]
    range_atr = frame["range_atr"].iloc[i]
    
    if pd.isna(atr) or pd.isna(range_atr) or not np.isfinite(atr):
        return None
    
    # Check si hubo compresión en los últimos N bares
    recent_range_atr = frame["range_atr"].iloc[i-compression_bars:i]
    if recent_range_atr.max() > compression_threshold:
        return None  # No hubo compresión
    
    # Calcular rango de compresión
    recent_high = frame["high"].iloc[i-compression_bars:i].max()
    recent_low = frame["low"].iloc[i-compression_bars:i].min()
    
    current_high = frame["high"].iloc[i]
    current_low = frame["low"].iloc[i]
    current_close = frame["close"].iloc[i]
    
    # Breakout hacia arriba
    if current_close > recent_high:
        return {
            "direction": "long",
            "stop_atr": atr_mult,
            "entry_price": current_close,
            "stop_mode": "atr",
            "target_rr": target_rr,
            "break_even_at_r": break_even_at_r,
        }
    
    # Breakout hacia abajo
    if current_close < recent_low:
        return {
            "direction": "short",
            "stop_atr": atr_mult,
            "entry_price": current_close,
            "stop_mode": "atr",
            "target_rr": target_rr,
            "break_even_at_r": break_even_at_r,
        }
    
    return None

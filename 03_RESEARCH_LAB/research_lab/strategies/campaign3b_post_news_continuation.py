"""
Campaign 3B - M3: Post-News Structured Continuation
Ventana: 08:00 - 11:00 NY
Timeframe: M15
Lógica: Continuación objetiva después del último high-impact relevante
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session

NAME = "campaign3b_post_news_continuation"
WARMUP_BARS = 50
EXPLICIT_TIMEFRAME = "M15"
PIP_SIZE = 0.0001

DEFAULT_GRID = [
    {
        "session_name": "am_08_11",
        "entry_start": "08:00",
        "entry_end": "11:00",
        "target_rr": 2.0,
        "break_even_at_r": 1.0,
        "max_hold_bars": 12,
        "cooldown_bars": 12,
        "atr_mult_sl": 1.5,
        "post_news_wait_bars": 2,
        "trend_ema_period": 20,
        "trend_slope_bars": 3,
    }
]

def parameter_space() -> dict[str, list]:
    return {
        "atr_mult_sl": [1.3, 1.5, 1.8],
        "post_news_wait_bars": [2, 3, 4],
        "trend_ema_period": [15, 20, 25],
        "trend_slope_bars": [2, 3, 4],
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
    
    session_name = params.get("session_name", "am_08_11")
    atr_mult = params.get("atr_mult_sl", 1.5)
    wait_bars = params.get("post_news_wait_bars", 2)
    ema_period = params.get("trend_ema_period", 20)
    slope_bars = params.get("trend_slope_bars", 3)
    target_rr = params.get("target_rr", 2.0)
    break_even_at_r = params.get("break_even_at_r", 1.0)
    
    # Verificar sesión AM
    if not is_in_session(frame.index[i], session_name):
        return None
    
    # Verificar que estamos fuera de news block (esperar wait_bars post-news)
    # Esto es manejado por el engine con news_block, pero podemos agregar lógica adicional
    
    atr = frame["atr14"].iloc[i]
    ema_col = f"ema{ema_period}"
    
    if ema_col not in frame.columns:
        return None
    
    ema = frame[ema_col].iloc[i]
    
    if pd.isna(atr) or pd.isna(ema) or not np.isfinite(atr):
        return None
    
    # Calcular slope de EMA
    if i < slope_bars:
        return None
    
    ema_slope = ema - frame[ema_col].iloc[i - slope_bars]
    
    current_close = frame["close"].iloc[i]
    
    # Tendencia alcista
    if ema_slope > 0:
        # Esperar pullback a EMA o debajo
        if current_close <= ema * 1.001:  # Pequeña tolerancia
            return {
                "direction": "long",
                "stop_atr": atr_mult,
                "entry_price": current_close,
                "stop_mode": "atr",
                "target_rr": target_rr,
                "break_even_at_r": break_even_at_r,
            }
    
    # Tendencia bajista
    if ema_slope < 0:
        # Esperar pullback a EMA o encima
        if current_close >= ema * 0.999:
            return {
                "direction": "short",
                "stop_atr": atr_mult,
                "entry_price": current_close,
                "stop_mode": "atr",
                "target_rr": target_rr,
                "break_even_at_r": break_even_at_r,
            }
    
    return None

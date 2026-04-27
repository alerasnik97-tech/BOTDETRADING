"""
Campaign 3B - M4: Session Expansion
Ventana: 08:00 - 11:00 NY
Timeframe: M15
Lógica: Expansión de sesión desde Asia a NY
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import is_in_session

NAME = "campaign3b_session_expansion"
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
        "cooldown_bars": 8,
        "atr_mult_sl": 1.5,
        "asia_ny_expansion_factor": 1.5,
        "confirm_bars": 1,
    }
]

def parameter_space() -> dict[str, list]:
    return {
        "atr_mult_sl": [1.3, 1.5, 1.8],
        "asia_ny_expansion_factor": [1.3, 1.5, 1.8],
        "confirm_bars": [1, 2],
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
    expansion_factor = params.get("asia_ny_expansion_factor", 1.5)
    confirm_bars = params.get("confirm_bars", 1)
    target_rr = params.get("target_rr", 2.0)
    break_even_at_r = params.get("break_even_at_r", 1.0)
    
    # Verificar sesión AM
    if not is_in_session(frame.index[i], session_name):
        return None
    
    # Niveles de session ranges
    asia_high_col = "session_range_high_00_00_07_00"
    asia_low_col = "session_range_low_00_00_07_00"
    
    if asia_high_col not in frame.columns or asia_low_col not in frame.columns:
        return None
    
    asia_high = frame[asia_high_col].iloc[i]
    asia_low = frame[asia_low_col].iloc[i]
    
    if pd.isna(asia_high) or pd.isna(asia_low):
        return None
    
    asia_range = asia_high - asia_low
    
    # Calcular rango de NY session hasta ahora (08:00 - current)
    # Asumimos que estamos en sesión AM
    ny_session_start = pd.Timestamp(frame.index[i].date()) + pd.Timedelta(hours=8, minutes=0)
    ny_session_start = ny_session_start.tz_localize(frame.index[i].tz)
    
    # Filtrar bares desde 08:00 hasta ahora
    mask = (frame.index[i-50:i] >= ny_session_start) & (frame.index[i-50:i] <= frame.index[i])
    ny_bars = frame.iloc[i-50:i][mask]
    
    if len(ny_bars) < 3:
        return None
    
    ny_high = ny_bars["high"].max()
    ny_low = ny_bars["low"].min()
    ny_range = ny_high - ny_low
    
    atr = frame["atr14"].iloc[i]
    
    if pd.isna(atr) or not np.isfinite(atr) or asia_range == 0:
        return None
    
    # Verificar expansión
    if ny_range < asia_range * expansion_factor:
        return None  # No hay expansión suficiente
    
    current_close = frame["close"].iloc[i]
    current_high = frame["high"].iloc[i]
    current_low = frame["low"].iloc[i]
    
    # Breakout hacia arriba de Asia range
    if i >= confirm_bars and current_close > asia_high:
        return {
            "direction": "long",
            "stop_atr": atr_mult,
            "entry_price": current_close,
            "stop_mode": "atr",
            "target_rr": target_rr,
            "break_even_at_r": break_even_at_r,
        }
    
    # Breakout hacia abajo de Asia range
    if i >= confirm_bars and current_close < asia_low:
        return {
            "direction": "short",
            "stop_atr": atr_mult,
            "entry_price": current_close,
            "stop_mode": "atr",
            "target_rr": target_rr,
            "break_even_at_r": break_even_at_r,
        }
    
    return None

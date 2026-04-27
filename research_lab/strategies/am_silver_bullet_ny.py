from __future__ import annotations

from typing import Any

import pandas as pd

from research_lab.ict_primitives import (
    add_fvg_columns,
    add_pivot_structure_columns,
    ensure_session_range_columns,
)
from research_lab.strategies.common import is_in_session


NAME = "am_silver_bullet_ny"
WARMUP_BARS = 120
EXPLICIT_TIMEFRAME = "M5"
PIP_SIZE = 0.0001


DEFAULT_GRID: list[dict[str, Any]] = [
    {
        "variant_label": "canonical_10_11",
        "session_name": "am_08_11",
        "entry_start": "10:00",
        "entry_end": "11:00",
        "anchor_start": "03:00",
        "anchor_end": "08:30",
        "target_rr": 2.0,
        "max_hold_bars": 12,
        "cooldown_bars": 8,
        "break_even_at_r": 1.0,
    }
]


def parameter_space() -> dict[str, list[Any]]:
    return {}


def parameter_grid(max_combinations: int = 1, seed: int = 42) -> list[dict[str, Any]]:
    del seed
    return DEFAULT_GRID[:max_combinations]


def default_params() -> dict[str, Any]:
    return dict(DEFAULT_GRID[0])


def _is_time_to_trade(ts: pd.Timestamp, params: dict[str, Any]) -> bool:
    minute_value = ts.hour * 60 + ts.minute
    start_h, start_m = map(int, str(params["entry_start"]).split(":"))
    end_h, end_m = map(int, str(params["entry_end"]).split(":"))
    return (start_h * 60 + start_m) <= minute_value < (end_h * 60 + end_m)


def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
    ts = frame.index[i]
    if i < 2 or not is_in_session(ts, str(params["session_name"])) or not _is_time_to_trade(ts, params):
        return None

    # Columnas de Anclaje
    high_col = "session_range_high_03_00_08_30"
    low_col = "session_range_low_03_00_08_30"

    if high_col not in frame.columns:
        return None 

    anchor_high = frame[high_col].iat[i]
    anchor_low = frame[low_col].iat[i]
    if not (anchor_high == anchor_high and anchor_low == anchor_low):
        return None

    # Columnas de estado (inyectadas o calculadas al vuelo)
    # Para backtest serio, deberiamos inyectarlas en el loader, 
    # pero aqui podemos usar un chequeo hacia atras para "Setup Activo"
    
    # Buscamos un setup (Sweep + MSS + FVG) en los ultimos 12 bares (1 hora en M5)
    setup = None
    lookback = 12
    for j in range(i - 1, max(0, i - lookback), -1):
        # Evitamos re-usar el mismo setup si ya paso mucho tiempo o cambio la estructura radicalmente
        # (Pero para SB, un setup detectado a las 10:05 sigue siendo valido a las 10:20 si no se ha invalidado)
        
        j_ts = frame.index[j]
        if not _is_time_to_trade(j_ts, params):
            continue

        has_swept_high = float(frame["day_running_high"].iat[j]) > (anchor_high + 0.0)
        has_swept_low = float(frame["day_running_low"].iat[j]) < (anchor_low - 0.0)
        
        bullish_mss = bool(frame["bullish_choch"].iat[j])
        bearish_mss = bool(frame["bearish_choch"].iat[j])
        
        # Long Setup check at j
        if has_swept_low and bullish_mss:
            for k in range(j, j - 4, -1):
                if k < 0: break
                if bool(frame["bullish_fvg"].iat[k]):
                    setup = {"direction": "long", "price": float(frame["bullish_fvg_mid"].iat[k]), "sl": anchor_low - 2.0 * PIP_SIZE}
                    break
        
        # Short Setup check at j
        if not setup and has_swept_high and bearish_mss:
            for k in range(j, j - 4, -1):
                if k < 0: break
                if bool(frame["bearish_fvg"].iat[k]):
                    setup = {"direction": "short", "price": float(frame["bearish_fvg_mid"].iat[k]), "sl": anchor_high + 2.0 * PIP_SIZE}
                    break
        
        if setup: break

    if setup:
        # Verificamos si la vela ACTUAL toca el precio de setup
        curr_low = float(frame["low"].iat[i])
        curr_high = float(frame["high"].iat[i])
        
        if setup["direction"] == "long":
            if curr_low <= setup["price"] <= curr_high:
                if setup["price"] <= setup["sl"]:
                    # print(f"DEBUG AM SB: Signal Long invalid SL at {ts}")
                    return None
                return {
                    "direction": "long",
                    "entry_mode": "limit",
                    "limit_price": setup["price"],
                    "stop_mode": "price",
                    "stop_price": setup["sl"],
                    "target_mode": "rr",
                    "target_rr": float(params["target_rr"]),
                    "max_hold_bars": int(params["max_hold_bars"]),
                    "cooldown_bars": int(params.get("cooldown_bars", 0)),
                    "break_even_at_r": params.get("break_even_at_r"),
                    "session_name": str(params["session_name"]),
                }
        else:
            if curr_low <= setup["price"] <= curr_high:
                if setup["price"] >= setup["sl"]:
                    # print(f"DEBUG AM SB: Signal Short invalid SL at {ts}")
                    return None
                return {
                    "direction": "short",
                    "entry_mode": "limit",
                    "limit_price": setup["price"],
                    "stop_mode": "price",
                    "stop_price": setup["sl"],
                    "target_mode": "rr",
                    "target_rr": float(params["target_rr"]),
                    "max_hold_bars": int(params["max_hold_bars"]),
                    "cooldown_bars": int(params.get("cooldown_bars", 0)),
                    "break_even_at_r": params.get("break_even_at_r"),
                    "session_name": str(params["session_name"]),
                }

    return None

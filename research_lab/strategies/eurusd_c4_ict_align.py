from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import add_general_params, stratified_sample_combinations
from research_lab.ict_primitives import find_recent_sweep_event

NAME = "eurusd_c4_ict_align"
WARMUP_BARS = 200
EXPLICIT_TIMEFRAME = "M5"

def parameter_space() -> dict[str, list]:
    return add_general_params(
        {
            "max_hold": [30],
            "cooldown_bars": [10],
            "sweep_penetration_pips": [0.5],
            "target_rr": [2.1],
            "max_age_sweep_bars": [50],
            "max_age_choch_bars": [30],
        }
    )

def parameter_grid(max_combinations: int = 1, seed: int = 42) -> list[dict]:
    params_dict = parameter_space()
    return stratified_sample_combinations(params_dict, max_combinations, seed)

def default_params() -> dict:
    return parameter_grid(1)[0]

def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Filtro horario estricto AM (08:00 - 11:00 NY)
    # El motor ya filtra la entrada, pero la logica de señal debe ser austera.
    
    # SHORT: Buscar confirmacion estructural (bearish_choch) y FVG primero
    if bool(frame["bearish_choch"].iat[i]) and bool(frame["bearish_fvg"].iat[i]):
        # Solo buscamos el sweep si ya tenemos CHoCH + FVG
        short_sweep = find_recent_sweep_event(
            frame, i,
            direction="short",
            min_penetration_pips=params["sweep_penetration_pips"],
            max_age_bars=params["max_age_sweep_bars"],
            level_columns=("prev_day_high", "london_high", "asia_high")
        )
        
        if short_sweep:
            fvg_top = float(frame["bearish_fvg_top"].iat[i])
            close_curr = float(frame["close"].iat[i])
            
            # Validar seguridad del Stop Loss (contra precio actual y contra el limit)
            stop_price = short_sweep.sweep_price + (1.0 * 0.0001)
            
            # Diferencial minimo de seguridad (0.2 pips)
            epsilon = 0.2 * 0.0001
            
            if np.isfinite(stop_price) and np.isfinite(fvg_top):
                if stop_price > close_curr + epsilon and stop_price > fvg_top + epsilon:
                    return {
                        "direction": "short",
                        "signal_price": close_curr,
                        "stop_mode": "price",
                        "stop_price": stop_price,
                        "target_mode": "rr",
                        "target_rr": params["target_rr"],
                        "max_hold_bars": params["max_hold"],
                        "cooldown_bars": params["cooldown_bars"],
                        "entry_mode": "limit",
                        "limit_price": fvg_top
                    }

    # LONG: Buscar confirmacion estructural (bullish_choch) y FVG primero
    if bool(frame["bullish_choch"].iat[i]) and bool(frame["bullish_fvg"].iat[i]):
        # Solo buscamos el sweep si ya tenemos CHoCH + FVG
        long_sweep = find_recent_sweep_event(
            frame, i,
            direction="long",
            min_penetration_pips=params["sweep_penetration_pips"],
            max_age_bars=params["max_age_sweep_bars"],
            level_columns=("prev_day_low", "london_low", "asia_low")
        )

        if long_sweep:
            fvg_bottom = float(frame["bullish_fvg_bottom"].iat[i])
            close_curr = float(frame["close"].iat[i])
            
            # Validar seguridad del Stop Loss (contra precio actual y contra el limit)
            stop_price = long_sweep.sweep_price - (1.0 * 0.0001)
            
            # Diferencial minimo de seguridad (0.2 pips)
            epsilon = 0.2 * 0.0001
            
            if np.isfinite(stop_price) and np.isfinite(fvg_bottom):
                if stop_price < close_curr - epsilon and stop_price < fvg_bottom - epsilon:
                    return {
                        "direction": "long",
                        "signal_price": close_curr,
                        "stop_mode": "price",
                        "stop_price": stop_price,
                        "target_mode": "rr",
                        "target_rr": params["target_rr"],
                        "max_hold_bars": params["max_hold"],
                        "cooldown_bars": params["cooldown_bars"],
                        "entry_mode": "limit",
                        "limit_price": fvg_bottom
                    }

    return None

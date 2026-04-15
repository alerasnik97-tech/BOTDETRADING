from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import add_general_params, stratified_sample_combinations


NAME = "strategy_src"
WARMUP_BARS = 150
EXPLICIT_TIMEFRAME = "M5"


def parameter_space() -> dict[str, list]:
    return add_general_params(
        {
            "max_hold": [24],
            "cooldown_bars": [5],
            "breakout_buffer_pips": [0.5],
            "retest_zone_pips": [1.0],
            "sl_buffer_pips": [1.0],
            "tp_atr_mult": [1.5],
            "be_atr_trigger": [1.0],
            "min_am_range_pips": [15.0],
            "max_retest_wait_bars": [18],
        }
    )


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    params_dict = parameter_space()
    # Forzar restricciones oficiales
    params_dict["session_name"] = ["light_fixed"]
    params_dict["use_h1_context"] = [False]
    params_dict["break_even_at_r"] = [1.0] # Usado para el BE disparado por ATR en esta implementacion
    return stratified_sample_combinations(params_dict, max_combinations, seed)


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Recuperar niveles AM (07:00 - 11:00)
    suffix = "11_00"
    max_am = frame[f"session_range_high_{suffix}"].iat[i]
    min_am = frame[f"session_range_low_{suffix}"].iat[i]
    am_complete = frame[f"session_range_complete_{suffix}"].iat[i]
    
    if not am_complete or np.isnan(max_am) or np.isnan(min_am):
        return None
        
    # 2. Filtro de Rango AM Minimo
    am_range = max_am - min_am
    if (am_range / 0.0001) < params["min_am_range_pips"]:
        return None
    
    am_midpoint = min_am + am_range * 0.5
    
    # 3. Datos de la vela actual (Gatillo)
    open_curr = float(frame["open"].iat[i])
    high_curr = float(frame["high"].iat[i])
    low_curr = float(frame["low"].iat[i])
    close_curr = float(frame["close"].iat[i])
    atr_curr = float(frame["atr14"].iat[i])
    
    breakout_buffer = params["breakout_buffer_pips"] * 0.0001
    retest_buffer = params["retest_zone_pips"] * 0.0001
    sl_buffer = params["sl_buffer_pips"] * 0.0001
    max_wait = params["max_retest_wait_bars"]
    
    # --- LOGICA LONG (Break de Max_AM) ---
    # Buscamos una ruptura en las ultimas N velas
    lookback = min(max_wait + 5, i) # Un poco mas por el 'outside'
    slice_df = frame.iloc[i-lookback : i]
    
    # Identificar la PRIMERA ruptura valida despues de las 11:00
    # Nota: El engine filtra por session_name, pero validamos hora por si acaso
    valid_breaks_long = slice_df[
        (slice_df["close"] > max_am + breakout_buffer) & 
        (slice_df.index.hour >= 11) & (slice_df.index.hour <= 15)
    ]
    
    if not valid_breaks_long.empty:
        # Tomamos el indice de la primera ruptura
        break_idx_relative = slice_df.index.get_loc(valid_breaks_long.index[0])
        break_idx_absolute = i - lookback + break_idx_relative
        
        # Verificar limites temporales (max_wait bars desde el break)
        if (i - break_idx_absolute) <= max_wait:
            # a. Verificar "desde afuera"
            # Al menos 1 vela completa por encima de Max_AM + Buffer
            after_break_slice = frame.iloc[break_idx_absolute + 1 : i+1]
            was_outside = any(after_break_slice["low"] > max_am + breakout_buffer)
            
            # b. Verificar no invalidacion estructural (Close < AM_Midpoint)
            not_invalidated = not any(after_break_slice["close"] < am_midpoint)
            
            # c. Verificar Retest y Gatillo en la vela actual (i)
            # Retest toca zona [Max_AM - 1pip, Max_AM + 1pip]
            touches_retest = low_curr <= (max_am + retest_buffer) and high_curr >= (max_am - retest_buffer)
            # Gatillo Alcista (Ruta B)
            is_bullish = close_curr > open_curr
            
            if was_outside and not_invalidated and touches_retest and is_bullish:
                return {
                    "direction": "long",
                    "stop_mode": "price",
                    "stop_price": low_curr - sl_buffer,
                    "target_mode": "price",
                    "target_price": close_curr + (atr_curr * params["tp_atr_mult"]),
                    "max_hold_bars": params["max_hold"],
                    "cooldown_bars": params["cooldown_bars"],
                    "session_name": params["session_name"],
                    "break_even_at_r": params["break_even_at_r"], # Usado por el motor como trigger de R
                    "stop_atr": params["tp_atr_mult"], # Guardado para referencia pero usamos stop_price directo
                }

    # --- LOGICA SHORT (Break de Min_AM) ---
    valid_breaks_short = slice_df[
        (slice_df["close"] < min_am - breakout_buffer) & 
        (slice_df.index.hour >= 11) & (slice_df.index.hour <= 15)
    ]
    
    if not valid_breaks_short.empty:
        break_idx_relative = slice_df.index.get_loc(valid_breaks_short.index[0])
        break_idx_absolute = i - lookback + break_idx_relative
        
        if (i - break_idx_absolute) <= max_wait:
            after_break_slice = frame.iloc[break_idx_absolute + 1 : i+1]
            was_outside = any(after_break_slice["high"] < min_am - breakout_buffer)
            not_invalidated = not any(after_break_slice["close"] > am_midpoint)
            
            touches_retest = high_curr >= (min_am - retest_buffer) and low_curr <= (min_am + retest_buffer)
            is_bearish = close_curr < open_curr
            
            if was_outside and not_invalidated and touches_retest and is_bearish:
                return {
                    "direction": "short",
                    "stop_mode": "price",
                    "stop_price": high_curr + sl_buffer,
                    "target_mode": "price",
                    "target_price": close_curr - (atr_curr * params["tp_atr_mult"]),
                    "max_hold_bars": params["max_hold"],
                    "cooldown_bars": params["cooldown_bars"],
                    "session_name": params["session_name"],
                    "break_even_at_r": params["break_even_at_r"],
                }

    return None

from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import add_general_params, stratified_sample_combinations


NAME = "strategy_ls_sr"
WARMUP_BARS = 120
EXPLICIT_TIMEFRAME = "M5"


def parameter_space() -> dict[str, list]:
    return add_general_params(
        {
            "max_hold": [24],
            "cooldown_bars": [5],
            "sweep_buffer_pips": [0.5],
            "sl_buffer_pips": [1.0],
            "min_am_range_pips": [15.0],
            "min_wick_ratio": [0.5],
        }
    )


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    # Forzamos las restricciones de la familia LS-SR V1.1
    params_dict = parameter_space()
    params_dict["session_name"] = ["light_fixed"]
    params_dict["use_h1_context"] = [False]
    return stratified_sample_combinations(params_dict, max_combinations, seed)


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # La señal se evalúa en i (en el cierre de la vela i)
    # Sin embargo, el motor llama a signal(frame, i, params) al final de la vela i
    # para entrar en i+1.
    
    # 1. Recuperar niveles AM (07:00 - 11:00)
    # Estos estan inyectados por data_loader en cada fila despues de las 11:00
    suffix = "11_00"
    max_am = frame[f"session_range_high_{suffix}"].iat[i]
    min_am = frame[f"session_range_low_{suffix}"].iat[i]
    am_complete = frame[f"session_range_complete_{suffix}"].iat[i]
    
    if not am_complete or np.isnan(max_am) or np.isnan(min_am):
        return None
        
    # 2. Filtro de Rango AM Minimo
    am_range_pips = (max_am - min_am) / 0.0001
    if am_range_pips < params["min_am_range_pips"]:
        return None
        
    # 3. Datos de la vela actual (i) que seria la vela de señal i-1 para la entrada i
    high_curr = float(frame["high"].iat[i])
    low_curr = float(frame["low"].iat[i])
    close_curr = float(frame["close"].iat[i])
    
    # Evitar division por cero (Vela flat)
    bar_range = high_curr - low_curr
    if bar_range <= 0:
        return None
        
    # 4. Target: 50% del rango AM
    target_price = min_am + (max_am - min_am) * 0.5
    
    sweep_buffer = params["sweep_buffer_pips"] * 0.0001
    sl_buffer = params["sl_buffer_pips"] * 0.0001
    min_wick_ratio = params["min_wick_ratio"]
    
    # --- LOGICA SHORT (Sweep de Max_AM) ---
    # a. Detectar Sweep
    is_sweep_short = high_curr > (max_am + sweep_buffer)
    # b. Detectar Rechazo (Cierre dentro)
    is_rejection_short = close_curr < max_am
    # c. Filtro de Intencion (Cierre en mitad inferior)
    wick_ratio_short = (high_curr - close_curr) / bar_range
    is_intention_short = wick_ratio_short >= min_wick_ratio
    
    if is_sweep_short and is_rejection_short and is_intention_short:
        # Validar que el precio no haya tocado el TP antes del setup
        # (Aunque el spec dice cancelar si toca TP antes del re-ingreso, 
        # aqui simplificamos asumiendo que el close_curr < max_am es la entrada)
        if close_curr > target_price: # Para un short, el precio debe estar por encima del target
            return {
                "direction": "short",
                "stop_mode": "price",
                "stop_price": high_curr + sl_buffer,
                "target_mode": "price",
                "target_price": target_price,
                "max_hold_bars": params["max_hold"],
                "cooldown_bars": params["cooldown_bars"],
                "session_name": params["session_name"],
            }
            
    # --- LOGICA LONG (Sweep de Min_AM) ---
    # a. Detectar Sweep
    is_sweep_long = low_curr < (min_am - sweep_buffer)
    # b. Detectar Rechazo (Cierre dentro)
    is_rejection_long = close_curr > min_am
    # c. Filtro de Intencion (Cierre en mitad superior)
    wick_ratio_long = (close_curr - low_curr) / bar_range
    is_intention_long = wick_ratio_long >= min_wick_ratio
    
    if is_sweep_long and is_rejection_long and is_intention_long:
        if close_curr < target_price:
            return {
                "direction": "long",
                "stop_mode": "price",
                "stop_price": low_curr - sl_buffer,
                "target_mode": "price",
                "target_price": target_price,
                "max_hold_bars": params["max_hold"],
                "cooldown_bars": params["cooldown_bars"],
                "session_name": params["session_name"],
            }
            
    return None

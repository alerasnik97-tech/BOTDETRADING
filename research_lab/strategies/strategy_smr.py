from __future__ import annotations

import pandas as pd
from research_lab.strategies.common import add_general_params, stratified_sample_combinations


NAME = "strategy_smr"
WARMUP_BARS = 120
EXPLICIT_TIMEFRAME = "M5"  # Pista para el loader


def parameter_space() -> dict[str, list]:
    params = add_general_params(
        {
            "bb_length": [20],
            "bb_std": [2.2],
            "rsi_length": [7],
            "rsi_extreme": [75],
            "sl_atr_mult": [1.5],
            "max_hold": [12],
            "cooldown_bars": [5],
        }
    )
    # Forzar restricciones del usuario
    params["session_name"] = ["light_fixed"]
    params["use_h1_context"] = [False]
    return params


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    # Para el baseline usamos los valores exactos definidos
    return stratified_sample_combinations(parameter_space(), max_combinations, seed)


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # La señal se evalúa en i-1 (datos cerrados)
    if i < 1:
        return None
    
    prev_i = i - 1
    
    # 1. Indicadores en i-1
    # Nota: Los nombres de columnas deben coincidir con prepare_common_frame
    suffix = "20_2_2"
    basis_prev = float(frame[f"bb_mid_{suffix}"].iat[prev_i])
    upper_prev = float(frame[f"bb_upper_{suffix}"].iat[prev_i])
    lower_prev = float(frame[f"bb_lower_{suffix}"].iat[prev_i])
    close_prev = float(frame["close"].iat[prev_i])
    rsi_prev = float(frame["rsi7"].iat[prev_i])
    atr_prev = float(frame["atr14"].iat[prev_i])
    
    # 2. Filtros de Señal en i-1
    dist_pips = abs(close_prev - basis_prev) / 0.0001
    band_width_pips = (upper_prev - lower_prev) / 0.0001
    
    if dist_pips <= 5.0:
        return None
    if band_width_pips <= 10.0:
        return None
        
    # 3. Lógica de Entrada (LONG)
    # Requisito V1.2: Close[i-1] < Lower[i-1] AND RSI[i-1] < 25 AND Open[i] < Lower[i-1]
    open_curr = float(frame["open"].iat[i])
    
    long_signal = (
        close_prev < lower_prev 
        and rsi_prev < 25 
        and open_curr < lower_prev
    )
    
    # 4. Lógica de Entrada (SHORT)
    # Requisito V1.2: Close[i-1] > Upper[i-1] AND RSI[i-1] > 75 AND Open[i] > Upper[i-1]
    short_signal = (
        close_prev > upper_prev 
        and rsi_prev > 75 
        and open_curr > upper_prev
    )
    
    if long_signal:
        return {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": params["sl_atr_mult"],
            "target_mode": "price",
            "target_price": basis_prev, # TP Fijo al valor de la media en el momento de señal
            "max_hold_bars": params["max_hold"],
            "cooldown_bars": params["cooldown_bars"],
            "session_name": params["session_name"],
        }
        
    if short_signal:
        return {
            "direction": "short",
            "stop_mode": "atr",
            "stop_atr": params["sl_atr_mult"],
            "target_mode": "price",
            "target_price": basis_prev,
            "max_hold_bars": params["max_hold"],
            "cooldown_bars": params["cooldown_bars"],
            "session_name": params["session_name"],
        }
        
    return None

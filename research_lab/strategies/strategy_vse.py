from __future__ import annotations

import pandas as pd
import numpy as np
from research_lab.strategies.common import add_general_params, stratified_sample_combinations


NAME = "strategy_vse"
WARMUP_BARS = 100
EXPLICIT_TIMEFRAME = "M5"


def parameter_space() -> dict[str, list]:
    return add_general_params(
        {
            "max_hold": [12],
            "cooldown_bars": [5],
            "bb_std": [2.0],
            "kc_mult": [1.5],
            "min_squeeze_bars": [5],
            "breakout_buffer_pips": [0.5],
            "limit_expiry_bars": [5],
            "tp_atr_mult": [0.8],
            "be_atr_trigger": [0.4],
            "body_pct_filter": [0.5],
        }
    )


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    params_dict = parameter_space()
    params_dict["session_name"] = ["light_fixed"]
    params_dict["use_h1_context"] = [False]
    params_dict["break_even_at_r"] = [0.0] # Not directly used as we check in strategy if possible, but engine uses it for BE
    return stratified_sample_combinations(params_dict, max_combinations, seed)


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    # 1. Indicadores necesarios (Se asume que estan en el frame o los calculamos)
    # Para ser robustos, los calculamos o usamos los nombres estandar
    # En VSE necesitamos BB y KC. 
    # Usaremos nombres calculados ad-hoc para evitar dependencias
    
    # BB (20, 2.0) -> bb_upper_20_2_0, bb_lower_20_2_0
    # KC (20, 1.5) -> kc_upper_20_1_5, kc_lower_20_1_5
    
    # 2. Verificar Ventana Horaria de Gatillo (13:00 - 16:00 NY)
    curr_hour = frame.index[i].hour
    if curr_hour < 13 or curr_hour >= 16:
        return None

    # 3. Datos de la vela actual
    high_i = float(frame["high"].iat[i])
    low_i = float(frame["low"].iat[i])
    close_i = float(frame["close"].iat[i])
    atr_i = float(frame["atr14"].iat[i])
    
    # 4. Buscar Breakout Valido en el pasado (hasta limit_expiry_bars)
    limit_expiry = params["limit_expiry_bars"]
    breakout_buffer = params["breakout_buffer_pips"] * 0.0001
    min_squeeze = params["min_squeeze_bars"]
    
    # Retrocedemos para buscar la vela de ruptura 'j'
    for lookback in range(1, limit_expiry + 1):
        j = i - lookback
        if j < 30: break
        
        bb_up_j = float(frame["bb_upper_20_2_0"].iat[j])
        bb_lo_j = float(frame["bb_lower_20_2_0"].iat[j])
        kc_up_j = float(frame["kc_upper_20_1_5"].iat[j])
        kc_lo_j = float(frame["kc_lower_20_1_5"].iat[j])
        
        in_squeeze_j = (bb_up_j < kc_up_j) and (bb_lo_j > kc_lo_j)
        
        # Squeeze Duration check
        squeeze_slice = (frame["bb_upper_20_2_0"].iloc[j-min_squeeze+1 : j+1] < frame["kc_upper_20_1_5"].iloc[j-min_squeeze+1 : j+1]) & \
                        (frame["bb_lower_20_2_0"].iloc[j-min_squeeze+1 : j+1] > frame["kc_lower_20_1_5"].iloc[j-min_squeeze+1 : j+1])
        
        if not squeeze_slice.all():
            continue
            
        # b. Ruptura en j
        close_j = float(frame["close"].iat[j])
        open_j = float(frame["open"].iat[j])
        high_j = float(frame["high"].iat[j])
        low_j = float(frame["low"].iat[j])
        
        # Filtro de primera ruptura (First Cross)
        close_prev_j = float(frame["close"].iat[j-1])
        bb_up_prev_j = float(frame["bb_upper_20_2_0"].iat[j-1])
        bb_lo_prev_j = float(frame["bb_lower_20_2_0"].iat[j-1])
        
        body_j = abs(close_j - open_j)
        range_j = high_j - low_j
        if range_j < 0.00001: continue
        
        is_break_long = (close_prev_j <= bb_up_prev_j + breakout_buffer) and \
                        (close_j > bb_up_j + breakout_buffer) and \
                        (body_j / range_j >= params["body_pct_filter"] - 1e-9)
                        
        is_break_short = (close_prev_j >= bb_lo_prev_j - breakout_buffer) and \
                         (close_j < bb_lo_j - breakout_buffer) and \
                         (body_j / range_j >= params["body_pct_filter"] - 1e-9)
        
        if not (is_break_long or is_break_short):
            continue
            
        # d. Verificar Cancelacion por Velocidad (TP alcanzado antes del fill)
        target_dist = 0.8 * float(frame["atr14"].iat[j])
        if is_break_long:
            tp_price = bb_up_j + target_dist 
            reached_tp = any(frame["high"].iloc[j+1 : i] >= tp_price)
            invalidated = any(frame["close"].iloc[j+1 : i] < bb_lo_j)
        else:
            tp_price = bb_lo_j - target_dist
            reached_tp = any(frame["low"].iloc[j+1 : i] <= tp_price)
            invalidated = any(frame["close"].iloc[j+1 : i] > bb_up_j)
            
        if reached_tp or invalidated:
            continue
            
        # f. GATILLO: Retest en vela actual 'i'
        if is_break_long:
            if low_i <= bb_up_j:
                return {
                    "direction": "long",
                    "stop_mode": "price",
                    "stop_price": bb_lo_j, # SL = BB Opp de la ruptura congelada
                    "target_mode": "price",
                    "target_price": bb_up_j + target_dist,
                    "max_hold_bars": params["max_hold"],
                    "cooldown_bars": params["cooldown_bars"],
                    "session_name": params["session_name"],
                    "break_even_at_r": 0.5, # 0.4 ATR / (Distancia SL) aprox. Usaremos params dinámicos si el motor lo soporta o BE manual
                    "signal_price": bb_up_j,
                }
        else:
            # Entry: Sell Limit en bb_lo_j
            if high_i >= bb_lo_j:
                return {
                    "direction": "short",
                    "stop_mode": "price",
                    "stop_price": bb_up_j,
                    "target_mode": "price",
                    "target_price": bb_lo_j - target_dist,
                    "max_hold_bars": params["max_hold"],
                    "cooldown_bars": params["cooldown_bars"],
                    "session_name": params["session_name"],
                    "break_even_at_r": 0.5,
                    "signal_price": bb_lo_j,
                }
                
    return None

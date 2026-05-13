
import pandas as pd
from dataclasses import dataclass
from .numeric import snap_to_tick

@dataclass
class FillResult:
    fill_time: pd.Timestamp
    fill_price: float
    side: str  # "long" | "short"
    signal_time: pd.Timestamp
    signal_bar_close: pd.Timestamp
    reason: str = "SIGNAL"
    slippage_pips: float = 0.0

class NoFillError(Exception): pass

def next_bar_execute(side: str, 
                     signal_bar_close: pd.Timestamp, 
                     ticks_after: pd.DataFrame, 
                     instrument: str = "EURUSD") -> FillResult:
    """
    Ejecuta en el primer tick disponible después del cierre de la vela de señal (T+1).
    """
    # Buscar primer tick con ts > signal_bar_close
    candidates = ticks_after[ticks_after.index > signal_bar_close]
    if candidates.empty:
        raise NoFillError(f"No hay ticks disponibles después de {signal_bar_close} (posible gap de fin de semana).")
    
    first_tick = candidates.iloc[0]
    
    if side == "long":
        price = first_tick["ask"]
    else:
        price = first_tick["bid"]
        
    return FillResult(
        fill_time=candidates.index[0],
        fill_price=snap_to_tick(price, instrument),
        side=side,
        signal_time=signal_bar_close, # Simplificación: señal ocurre al cierre
        signal_bar_close=signal_bar_close
    )

def simulate_exit(side: str, 
                  entry_price: float, 
                  sl_price: float, 
                  tp_price: float, 
                  ticks_during: pd.DataFrame, 
                  fill_time: pd.Timestamp,
                  time_exit: pd.Timestamp | None = None,
                  instrument: str = "EURUSD") -> FillResult:
    """
    Simulación tick-a-tick de la salida (SL/TP/Time).
    Resuelve ambigüedad intra-vela.
    D1 FIX: Salta ticks con index <= fill_time.
    """
    ticks_eval = ticks_during[ticks_during.index > fill_time]
    if ticks_eval.empty:
        if time_exit and fill_time >= time_exit:
            # Caso especial: fill ocurrió justo en o después del time_exit
             return FillResult(fill_time, entry_price, side, fill_time, fill_time, "TIME")
        raise NoFillError(f"No hay ticks post-fill para {fill_time}")

    for ts, row in ticks_eval.iterrows():
        if side == "long":
            # Salida Long es en el Bid
            current_price = row["bid"]
            if current_price <= sl_price:
                return FillResult(ts, sl_price, "long", ts, ts, "SL")
            if current_price >= tp_price:
                return FillResult(ts, tp_price, "long", ts, ts, "TP")
        else:
            # Salida Short es en el Ask
            current_price = row["ask"]
            if current_price >= sl_price:
                return FillResult(ts, sl_price, "short", ts, ts, "SL")
            if current_price <= tp_price:
                return FillResult(ts, tp_price, "short", ts, ts, "TP")
                
        if time_exit and ts >= time_exit:
            return FillResult(ts, current_price, side, ts, ts, "TIME")
            
    # Si termina el dataframe sin tocar nada, salir al último precio disponible
    last_tick = ticks_eval.iloc[-1]
    return FillResult(ticks_eval.index[-1], 
                      last_tick["bid"] if side=="long" else last_tick["ask"], 
                      side, ticks_eval.index[-1], ticks_eval.index[-1], "EOM")

def simulate_exit_with_be(side: str, 
                          entry_price: float, 
                          sl_price: float, 
                          be_trigger: float, 
                          be_new_sl: float, 
                          tp_price: float, 
                          ticks_during: pd.DataFrame, 
                          fill_time: pd.Timestamp,
                          time_exit: pd.Timestamp | None = None,
                          instrument: str = "EURUSD") -> FillResult:
    """
    Simulación tick-a-tick con lógica de Break-Even (BE).
    B.2 Mandato V6.3.5.
    """
    ticks_eval = ticks_during[ticks_during.index > fill_time]
    if ticks_eval.empty:
        raise NoFillError(f"No hay ticks post-fill para {fill_time}")

    active_sl = sl_price
    be_armed = False

    for ts, row in ticks_eval.iterrows():
        if side == "long":
            price = row["bid"]
            # Trigger BE
            if not be_armed and price >= be_trigger:
                be_armed = True
                active_sl = be_new_sl
            
            # Check Exit
            if price <= active_sl:
                reason = "BE-SL" if be_armed else "SL"
                return FillResult(ts, active_sl, "long", ts, ts, reason)
            if price >= tp_price:
                return FillResult(ts, tp_price, "long", ts, ts, "TP")
        else:
            price = row["ask"]
            # Trigger BE
            if not be_armed and price <= be_trigger:
                be_armed = True
                active_sl = be_new_sl
            
            # Check Exit
            if price >= active_sl:
                reason = "BE-SL" if be_armed else "SL"
                return FillResult(ts, active_sl, "short", ts, ts, reason)
            if price <= tp_price:
                return FillResult(ts, tp_price, "short", ts, ts, "TP")

        if time_exit and ts >= time_exit:
            return FillResult(ts, price, side, ts, ts, "TIME")

    last_tick = ticks_eval.iloc[-1]
    return FillResult(ticks_eval.index[-1], 
                      last_tick["bid"] if side=="long" else last_tick["ask"], 
                      side, ticks_eval.index[-1], ticks_eval.index[-1], "EOM")

def next_bar_execute_limit(side: str,
                            signal_time: pd.Timestamp,
                            limit_price: float,
                            ticks_after: pd.DataFrame,
                            tif_minutes: int = 30,
                            instrument: str = "EURUSD"
                            ) -> FillResult | None:
    """
    Limit order con TIF.
    
    Para LONG (buy limit POR DEBAJO del precio actual):
        Fill cuando ASK <= limit_price.
        Fill price = limit_price (broker honra limit exacto).
    
    Para SHORT (sell limit POR ENCIMA del precio actual):
        Fill cuando BID >= limit_price.
        Fill price = limit_price.
    
    Expiry: signal_time + tif_minutes. Si nunca se toca → None.
    
    CRÍTICO: usar ASK para buy limit (porque en MT5 real, la orden
    de compra se llena al ask). NO usar mid ni bid. La diferencia
    es ~0.4 pips de fricción real que muchos backtests omiten.
    """
    expiry = signal_time + pd.Timedelta(minutes=tif_minutes)
    window = ticks_after[(ticks_after.index > signal_time) & 
                         (ticks_after.index <= expiry)]
    if window.empty:
        return None
    
    if side == "long":
        crossed = window[window["ask"] <= limit_price]
    else:
        crossed = window[window["bid"] >= limit_price]
    
    if crossed.empty:
        return None
    
    return FillResult(
        fill_time=crossed.index[0],
        fill_price=snap_to_tick(limit_price, instrument),
        side=side,
        signal_time=signal_time,
        signal_bar_close=signal_time,
        reason="LIMIT_FILL"
    )

def next_bar_execute_stop(
    side: str,
    signal_bar_close: pd.Timestamp,
    stop_price: float,
    ticks_after: pd.DataFrame,
    expiry_minutes: int = 60,
    instrument: str = "EURUSD"
) -> FillResult | None:
    """
    Ejecuta una orden Stop real condicionada al cruce causal de la cotización.
    
    Para LONG: Stop entry por encima del nivel predefinido; se llena con el Ask si Ask >= stop_price.
    Para SHORT: Stop entry por debajo del nivel predefinido; se llena con el Bid si Bid <= stop_price.
    """
    expiry = signal_bar_close + pd.Timedelta(minutes=expiry_minutes)
    candidates = ticks_after[(ticks_after.index > signal_bar_close) & (ticks_after.index <= expiry)]
    if candidates.empty:
        return None
        
    if side == "long":
        crossed = candidates[candidates["ask"] >= stop_price]
        if crossed.empty:
            return None
        first_cross = crossed.iloc[0]
        fill_p = max(stop_price, float(first_cross["ask"]))
        fill_t = crossed.index[0]
    else:
        crossed = candidates[candidates["bid"] <= stop_price]
        if crossed.empty:
            return None
        first_cross = crossed.iloc[0]
        fill_p = min(stop_price, float(first_cross["bid"]))
        fill_t = crossed.index[0]
        
    return FillResult(
        fill_time=fill_t,
        fill_price=snap_to_tick(fill_p, instrument),
        side=side,
        signal_time=signal_bar_close,
        signal_bar_close=signal_bar_close,
        reason="STOP_FILL"
    )


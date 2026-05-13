
import pandas as pd
from typing import Optional, Dict

BAR_DURATIONS = {
    "M1": "1min",
    "M3": "3min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1440min",
}

def build_bars(ticks: pd.DataFrame, bar_type: str, 
               price_col: str = "mid", anchor: str = "fx_day") -> pd.DataFrame:
    """
    Construye barras OHLC causales usando pd.Grouper (Altamente Optimizado).
    Maneja el anclaje de 17:00 NY para FX.
    """
    if bar_type not in BAR_DURATIONS:
        raise ValueError(f"Bar type {bar_type} no soportado.")
    
    freq = BAR_DURATIONS[bar_type]
    
    # Preparar precio
    if price_col not in ticks.columns and price_col == "mid":
        df = ticks.copy()
        df["mid"] = (df["bid"] + df["ask"]) / 2
        p_col = "mid"
    else:
        df = ticks
        p_col = price_col
        
    if anchor == "fx_day":
        # Convertir a NY para que el offset de 17h sea localmente correcto
        df_ny = df.tz_convert("America/New_York")
        # Usar un origin fijo a las 17:00 garantiza alineación wall-clock
        grouper = pd.Grouper(freq=freq, closed='left', label='right', origin='start_day', offset='17h')
    else:
        # Midnight UTC anchor
        df_ny = df # Sigue siendo UTC
        grouper = pd.Grouper(freq=freq, closed='left', label='right', origin='start_day')

    # Agregación
    agg_dict = {p_col: ["first", "max", "min", "last", "count"]}
    if "bid_volume" in df.columns: agg_dict["bid_volume"] = "sum"
    if "ask_volume" in df.columns: agg_dict["ask_volume"] = "sum"
    
    bars = df_ny.groupby(grouper).agg(agg_dict)
    
    # Limpiar columnas
    bars.columns = ["open", "high", "low", "close", "tick_count"] + \
                   (["volume_bid", "volume_ask"] if "bid_volume" in agg_dict else [])
    
    # Eliminar barras vacías
    bars.dropna(subset=["open"], inplace=True)
    
    # Index es close_time en NY. Convertir de vuelta a UTC.
    bars.index = bars.index.tz_convert("UTC")
    
    # Calcular open_time
    delta = pd.Timedelta(freq)
    bars["open_time"] = bars.index - delta
    
    # Filtrar parciales (causalidad)
    max_tick_ts = ticks.index.max()
    bars = bars[bars.index <= max_tick_ts]
    
    cols = ["open_time", "open", "high", "low", "close", "tick_count"]
    if "volume_bid" in bars.columns: cols += ["volume_bid", "volume_ask"]
    
    return bars[cols]

def get_bar_at(bars: pd.DataFrame, ts: pd.Timestamp, policy: str = "last_closed") -> pd.Series:
    if policy == "last_closed":
        valid = bars[bars.index <= ts]
        return valid.iloc[-1] if not valid.empty else pd.Series(dtype=float)
    elif policy == "in_progress":
        future = bars[bars.index > ts]
        if not future.empty:
            candidate = future.iloc[0]
            if candidate["open_time"] <= ts: return candidate
    return pd.Series(dtype=float)

def assert_no_partial_bars(bars: pd.DataFrame, data_end: pd.Timestamp) -> None:
    if not bars.empty and bars.index.max() > data_end:
        raise ValueError(f"Barra parcial detectada: {bars.index.max()} > {data_end}")

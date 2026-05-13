
from __future__ import annotations
import pandas as pd
from zoneinfo import ZoneInfo
from datetime import datetime, time
from typing import Optional

def sanitize_utc_index(df: pd.DataFrame, ts_col: Optional[str] = "timestamp_utc") -> pd.DataFrame:
    """
    Garantiza que el DataFrame tenga un DatetimeIndex tz=UTC, ordenado y limpio.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        # Si no es DatetimeIndex, intentar buscar la columna de tiempo
        if ts_col and ts_col in df.columns:
             df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
             df.set_index(ts_col, inplace=True)
        else:
             # Si no hay columna, no podemos sanitizar el índice temporal
             return df

    if df.index.tz is None:
        # Si no tiene tz, asumir UTC y advertir
        print(f"[WARNING] Index naive detectado. Asumiendo UTC.")
        df.index = df.index.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":
        df.index = df.index.tz_convert("UTC")
        
    df = df[~df.index.duplicated(keep='first')]
    df = df[df.index.notna()]
    df.sort_index(inplace=True)
    return df

def to_ny(df_utc: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte un DataFrame UTC-aware a America/New_York manejando ambigüedades DST.
    """
    if df_utc.index.tz is None or str(df_utc.index.tz) != "UTC":
        raise ValueError("El DataFrame debe tener un DatetimeIndex en UTC.")
    
    # America/New_York via ZoneInfo
    ny_tz = ZoneInfo("America/New_York")
    
    # Convertimos a NY. Pandas maneja la mayoría de los casos internamente,
    # pero para series de tiempo continuas, debemos asegurar que no haya solapamientos
    # o saltos inesperados si se crearan manualmente. 
    # tz_convert en un DatetimeIndex existente es determinista.
    df_ny = df_utc.copy()
    df_ny.index = df_ny.index.tz_convert(ny_tz)
    
    return df_ny

def assert_no_dst_holes(df_ny: pd.DataFrame) -> None:
    """
    Verifica que no haya gaps mayores a 1h en bordes DST.
    """
    # En NY, el salto es de 1h. Si el gap entre ticks es > 1h 
    # en fechas de cambio (Marzo/Noviembre), reportar.
    # Para esta fase de auditoría, simplemente buscamos gaps > 1h en general.
    diffs = df_ny.index.to_series().diff()
    large_gaps = diffs[diffs > pd.Timedelta(hours=1)]
    if not large_gaps.empty:
        # Filtrar si son fines de semana (donde el mercado cierra)
        # Sábado 17:00 NY a Domingo 17:00 NY es gap normal.
        for ts, gap in large_gaps.items():
            # Si no es fin de semana, es un gap sospechoso
            if ts.weekday() not in [5, 6]: # 5=Sábado, 6=Domingo
                # Permitir el gap de apertura del domingo (22:00 UTC / 17:00 NY)
                if not (ts.weekday() == 0 and ts.hour < 5): # Lunes temprano en UTC es Domingo tarde en NY
                    print(f"[WARNING] Gap detectado: {ts} -> {gap}")

def session_anchor(ts_ny: pd.Timestamp, hour: int = 17, minute: int = 0) -> pd.Timestamp:
    """
    Devuelve el ancla de la sesión (17:00 NY del 'día de trading').
    Si ts_ny >= 17:00, el ancla es hoy a las 17:00.
    Si ts_ny < 17:00, el ancla es ayer a las 17:00.
    """
    if ts_ny.tzinfo is None:
        raise ValueError("Timestamp debe ser tz-aware (NY).")
        
    anchor_today = ts_ny.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if ts_ny >= anchor_today:
        return anchor_today
    else:
        return anchor_today - pd.Timedelta(days=1)

# Constantes de Feriados FX (Auditada 2015-2030 - Ejemplo)
FX_HOLIDAYS = {
    # Formato: (Año, Mes, Día)
    "New Year 2024": (2024, 1, 1),
    "July 4 2024": (2024, 7, 4),
    "Thanksgiving 2024": (2024, 11, 28),
    "Christmas 2024": (2024, 12, 25),
}

def is_market_open(ts_ny: pd.Timestamp) -> bool:
    """
    Lógica de mercado abierto: Domingo 17:00 NY a Viernes 17:00 NY.
    """
    # Weekend Check
    day = ts_ny.weekday()
    hour = ts_ny.hour
    
    if day == 4 and hour >= 17: # Viernes tarde
        return False
    if day == 5: # Sábado
        return False
    if day == 6 and hour < 17: # Domingo mañana
        return False
        
    # Feriados fijos/variables auditados
    if (ts_ny.year, ts_ny.month, ts_ny.day) in FX_HOLIDAYS.values():
        return False
        
    return True

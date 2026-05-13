
import os
import pandas as pd
import gc
from pathlib import Path
from typing import Iterator, Optional, List, Tuple
from .temporal import sanitize_utc_index
from .memory import MemoryGuard, safe_collect

PARQUET_ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA\tick\EURUSD\monthly")
DEFAULT_TICK_COLS = ["timestamp_utc", "bid", "ask"]
DEFAULT_TICK_COLS_FLOW = DEFAULT_TICK_COLS + ["bid_volume", "ask_volume"]

def iter_months(start: str, end: str) -> Iterator[Tuple[int, int]]:
    """
    Rinde tuplas (year, month) entre start y end (YYYY-MM).
    """
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    
    current = start_ts.replace(day=1)
    while current <= end_ts:
        yield (current.year, current.month)
        current = (current + pd.DateOffset(months=1))

def parquet_path_for(year: int, month: int) -> Path:
    """
    Retorna la ruta al parquet mensual.
    """
    filename = f"EURUSD_ticks_{year}_{month:02d}.parquet"
    path = PARQUET_ROOT / filename
    if not path.exists():
        raise FileNotFoundError(f"Parquet no encontrado: {path}")
    return path

def load_month(year: int, month: int, 
               columns: Optional[List[str]] = None,
               downcast_floats: bool = True,
               sanitize: bool = True) -> pd.DataFrame:
    """
    Carga un mes de ticks con optimizaciones.
    """
    if sanitize and columns is not None and "timestamp_utc" not in columns:
        columns = columns + ["timestamp_utc"]
        
    path = parquet_path_for(year, month)
    df = pd.read_parquet(path, columns=columns)
    
    if downcast_floats:
        # Downcast float64 -> float32 para bid/ask
        for col in ["bid", "ask", "bid_volume", "ask_volume"]:
            if col in df.columns:
                df[col] = df[col].astype("float32")
                
    if sanitize:
        df = sanitize_utc_index(df)
        
    return df

def iter_ticks_chunked(start: str, end: str,
                       columns: Optional[List[str]] = None,
                       chunk_months: int = 1,
                       sanitize: bool = True,
                       downcast_floats: bool = True) -> Iterator[pd.DataFrame]:
    """
    Generador que carga data por chunks de meses para optimizar RAM.
    """
    months = list(iter_months(start, end))
    
    for i in range(0, len(months), chunk_months):
        chunk_list = []
        for year, month in months[i:i+chunk_months]:
            df_month = load_month(year, month, columns, downcast_floats, sanitize)
            chunk_list.append(df_month)
            
        if len(chunk_list) == 1:
            yield chunk_list[0]
        else:
            yield pd.concat(chunk_list)
            
        # Cleanup explícito tras yield
        del chunk_list
        safe_collect()

def load_range_bulk(start: str, end: str,
                    columns: Optional[List[str]] = None,
                    max_budget_mb: int = 4096) -> pd.DataFrame:
    """
    Carga un rango completo a RAM bajo protección de MemoryGuard.
    """
    with MemoryGuard(budget_mb=max_budget_mb, label="load_range_bulk") as guard:
        chunks = []
        for df in iter_ticks_chunked(start, end, columns=columns, sanitize=True, downcast_floats=True):
            chunks.append(df)
            guard.check()
            
        return pd.concat(chunks)

def estimate_chunk_rss_mb(columns: Optional[List[str]] = None, 
                          downcast: bool = True) -> float:
    """
    Estimador heurístico basado en A.3.
    """
    cols = columns or ["timestamp_utc", "bid", "ask", "bid_volume", "ask_volume"]
    base_per_col = 20.0 # MB promedio por columna float64 al mes
    if downcast:
        base_per_col = 13.0
        
    return len(cols) * base_per_col


import pandas as pd
import traceback
from datetime import datetime
from typing import Optional, Any

class LookAheadError(Exception):
    """Error disparado cuando un backtest intenta leer data del futuro."""
    pass

class CausalClock:
    def __init__(self, start_ts: pd.Timestamp):
        self.now = start_ts
        self.permissive = False

    def advance_to(self, ts: pd.Timestamp):
        if ts < self.now:
            raise ValueError(f"El reloj no puede retroceder: {ts} < {self.now}")
        self.now = ts
        
    def tick(self, ts: pd.Timestamp):
        self.advance_to(ts)

class CausalLog:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CausalLog, cls).__new__(cls)
            cls._instance.logs = []
        return cls._instance
    
    def log_access(self, ts_requested: pd.Timestamp, clock_now: pd.Timestamp, status: str):
        stack = traceback.extract_stack()
        # Buscar la línea que llamó al DataFrame (fuera de causal.py)
        caller = "Unknown"
        for frame in reversed(stack):
            if "causal.py" not in frame.filename and "pandas" not in frame.filename:
                caller = f"{frame.filename}:{frame.lineno} ({frame.name})"
                break
                
        self.logs.append({
            "requested": ts_requested,
            "now": clock_now,
            "status": status,
            "caller": caller
        })
        
    def clear(self):
        self.logs = []

class _CausalLocIndexer:
    def __init__(self, df: pd.DataFrame, clock: CausalClock, logger: CausalLog):
        self._df = df
        self._clock = clock
        self._logger = logger

    def __getitem__(self, key):
        # Interpretación básica del key para loguear
        requested_ts = None
        if isinstance(key, (pd.Timestamp, str, datetime)):
            requested_ts = pd.Timestamp(key)
        elif isinstance(key, slice) and key.stop:
            requested_ts = pd.Timestamp(key.stop)
        elif isinstance(key, tuple):
            # Caso loc[row_key, col_key]
            row_key = key[0]
            if isinstance(row_key, (pd.Timestamp, str, datetime)):
                requested_ts = pd.Timestamp(row_key)
            elif isinstance(row_key, slice) and row_key.stop:
                requested_ts = pd.Timestamp(row_key.stop)

        if requested_ts:
            # D3 FIX: Asegurar consistencia de timezone antes de comparar
            if requested_ts.tzinfo is None and self._clock.now.tzinfo is not None:
                requested_ts = requested_ts.tz_localize(self._clock.now.tzinfo)
            elif requested_ts.tzinfo is not None and self._clock.now.tzinfo is None:
                requested_ts = requested_ts.tz_localize(None)

            status = "OK" if requested_ts <= self._clock.now else "FUTURE_ERROR"
            self._logger.log_access(requested_ts, self._clock.now, status)
            if status == "FUTURE_ERROR" and not self._clock.permissive:
                raise LookAheadError(f"Look-Ahead detectado en .loc: {requested_ts} > {self._clock.now}")
        else:
            # Acceso genérico (ej: slice completo)
            self._logger.log_access(self._clock.now, self._clock.now, "OK_GENERIC")

        # Retornar el slice filtrado causalmente
        subset = self._df.loc[key]
        if isinstance(subset, pd.DataFrame):
            return subset[subset.index <= self._clock.now]
        elif isinstance(subset, pd.Series):
            if subset.name and isinstance(subset.name, pd.Timestamp) and subset.name > self._clock.now:
                 raise LookAheadError(f"Look-Ahead detectado en .loc (Series): {subset.name} > {self._clock.now}")
            return subset
        return subset

class CausalDataFrame:
    """
    Wrapper que intercepta accesos a un DataFrame para validar causalidad.
    D3 FIX: Harness activo vía _CausalLocIndexer y logging en get_causal.
    """
    def __init__(self, df: pd.DataFrame, clock: CausalClock):
        self._df = df
        self._clock = clock
        self._logger = CausalLog()

    def __getattr__(self, name):
        return getattr(self._df, name)

    @property
    def iloc(self):
        # iloc es difícil de validar causalmente sin conocer el mapeo index->pos
        # Para esta fase, logueamos acceso genérico
        self._logger.log_access(self._clock.now, self._clock.now, "ILOC_ACCESS")
        return self._df.iloc

    @property
    def loc(self):
        return _CausalLocIndexer(self._df, self._clock, self._logger)

    def get_causal(self):
        """Retorna el subset del dataframe visible al 'now' actual."""
        self._logger.log_access(self._clock.now, self._clock.now, "GET_CAUSAL")
        return self._df[self._df.index <= self._clock.now]

    def check_ts(self, ts: pd.Timestamp):
        if ts > self._clock.now:
            self._logger.log_access(ts, self._clock.now, "FUTURE_ERROR")
            if not self._clock.permissive:
                raise LookAheadError(f"Intento de leer data futura: {ts} > {self._clock.now}")
        else:
            self._logger.log_access(ts, self._clock.now, "OK")

def causal_audit(func):
    """Decorator para validar la firma de funciones de señal."""
    def wrapper(*args, **kwargs):
        # Verificar que se pasa el reloj
        has_clock = any(isinstance(a, CausalClock) for a in args) or "clock" in kwargs
        if not has_clock:
            print("[WARNING] Signal function llamada sin CausalClock.")
        return func(*args, **kwargs)
    return wrapper

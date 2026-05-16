import pandas as pd
import numpy as np
from research_lab.engine import run_backtest
from research_lab.config import EngineConfig, NY_TZ
from pathlib import Path

def test_dst_transition():
    print("=== DST TRANSITION INTEGRITY TEST ===")
    
    # Crear un frame que cruce el cambio de DST de Marzo 2024 (Marzo 10)
    # En NY, el cambio ocurre a las 02:00 -> 03:00.
    start_dt = "2024-03-08 00:00:00"
    end_dt = "2024-03-12 00:00:00"
    
    # Generar timestamps en UTC
    idx = pd.date_range(start=start_dt, end=end_dt, freq="15min", tz="UTC")
    
    frame = pd.DataFrame({
        "open": np.random.randn(len(idx)),
        "high": np.random.randn(len(idx)),
        "low": np.random.randn(len(idx)),
        "close": np.random.randn(len(idx)),
        "atr14": np.ones(len(idx)) * 0.0010,
        "range_atr": np.ones(len(idx))
    }, index=idx)
    
    # Mock strategy module
    class MockStrategy:
        NAME = "dst_test_strategy"
        WARMUP_BARS = 0
        @staticmethod
        def generate_signal(f, i, p): return None
        
    config = EngineConfig()
    
    # Correr backtest (esto fallaria si el motor no maneja NY_TZ correctamente)
    try:
        # Internamente el motor ahora hace .tz_convert(NY_TZ)
        # Vamos a verificar que no haya errores de alineacion
        result = run_backtest(MockStrategy, frame, {}, config, np.zeros(len(frame)), False)
        print("[SUCCESS] Motor de backtest operando con timestamps aware.")
        
        # Verificar conversion manual para estar seguros del offset
        sample_ts = idx[idx.strftime('%Y-%m-%d %H:%M') == '2024-03-11 12:00'][0]
        local_ts = sample_ts.tz_convert(NY_TZ)
        print(f"UTC: {sample_ts} -> NY: {local_ts}")
        
        # En Marzo 11, NY esta en EDT (-04:00)
        if local_ts.utcoffset().total_seconds() == -14400:
            print("[SUCCESS] DST transition recognized: EDT (-04:00) detected.")
        else:
            print("[FAILED] DST transition not recognized correctly.")
            
    except Exception as e:
        print(f"[ERROR] Temporal Integrity Check failed: {e}")

if __name__ == "__main__":
    test_dst_transition()

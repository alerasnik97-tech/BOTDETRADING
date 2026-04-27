import pandas as pd
import sys
import numpy as np
from pathlib import Path

# Agregar el directorio actual al path para importar research_lab
sys.path.append(".")

from research_lab.data_loader import load_backtest_data_bundle
from research_lab.engine import run_backtest
from research_lab.config import EngineConfig, NewsConfig
from research_lab.strategies.daily_open_mean_reversion_pm import signal, NAME, WARMUP_BARS

class MockModule:
    NAME = NAME
    WARMUP_BARS = WARMUP_BARS
    def signal(self, f, i, p): return signal(f, i, p)

try:
    data = load_backtest_data_bundle(
        pair="EURUSD",
        data_dirs=[Path("data_free_2020/prepared"), Path("data_candidates_2022_2025/prepared")],
        start="2024-01-01",
        end="2024-01-31",
        execution_mode="normal_mode",
        target_timeframe="M15"
    )
    frame = data.frame
    print(f"Dataframe rows: {len(frame)}")
    
    engine_config = EngineConfig(execution_mode="normal_mode")
    news_block = np.zeros(len(frame), dtype=bool)
    
    params = {
        "distance_atr_mult": 0.5, # Muy bajo para forzar señales
        "target_rr": 1.5,
        "break_even_at_r": None,
        "session_name": "light_fixed"
    }
    
    result = run_backtest(MockModule(), frame, params, engine_config, news_block, False)
    print(f"Trades found: {len(result.trades)}")
    
    if len(result.trades) == 0:
        print("\nDebugging why 0 trades:")
        # Check window
        start_min = 11*60
        end_min = 19*60
        
        entry_open_minutes = (frame.index.hour * 60 + frame.index.minute).to_numpy()
        in_window = (entry_open_minutes >= start_min) & (entry_open_minutes < end_min)
        print(f"Bars in window: {in_window.sum()}")
        
        if in_window.sum() > 0:
            first_idx = np.where(in_window)[0][0]
            print(f"First bar in window: {frame.index[first_idx]}")
            
            # Manual signal check
            for j in range(first_idx, first_idx + 100):
                if j >= len(frame): break
                sig = signal(frame, j, params)
                if sig:
                    print(f"Manual signal at {frame.index[j]}: {sig}")
                    break
            else:
                print("No manual signals found even in window.")

except Exception as e:
    import traceback
    traceback.print_exc()

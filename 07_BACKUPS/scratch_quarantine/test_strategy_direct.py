import pandas as pd
import sys
import numpy as np
from pathlib import Path

# Agregar el directorio actual al path para importar research_lab
sys.path.append(".")

from research_lab.data_loader import load_backtest_data_bundle
from research_lab.engine import run_backtest
from research_lab.config import EngineConfig, NewsConfig
from research_lab.strategies.prev_day_sweep_reversion_pm import signal, NAME, WARMUP_BARS

class MockModule:
    NAME = NAME
    WARMUP_BARS = WARMUP_BARS
    def signal(self, f, i, p): return signal(f, i, p)

try:
    data = load_backtest_data_bundle(
        pair="EURUSD",
        data_dirs=[Path("data_free_2020/prepared"), Path("data_candidates_2022_2025/prepared")],
        start="2020-01-01",
        end="2024-12-31",
        execution_mode="normal_mode",
        target_timeframe="M15"
    )
    frame = data.frame
    print(f"Dataframe rows: {len(frame)}")
    
    engine_config = EngineConfig(execution_mode="normal_mode")
    news_block = np.zeros(len(frame), dtype=bool)
    
    params = {
        "sweep_buffer_pips": 1.0, # 1 pip de margen
        "target_rr": 2.0,
        "break_even_at_r": 1.0,
        "session_name": "light_fixed"
    }
    
    result = run_backtest(MockModule(), frame, params, engine_config, news_block, False)
    print(f"Trades found: {len(result.trades)}")
    
    if not result.trades.empty:
        print("\nFirst 5 trades:")
        print(result.trades[["entry_time", "direction", "entry_price", "exit_price", "pnl_r"]].head(5))
    else:
        # Check reasons for no trades
        # 1. Look for signals manually
        signals = 0
        for i in range(WARMUP_BARS, len(frame)):
            sig = signal(frame, i, params)
            if sig:
                signals += 1
                if signals <= 5:
                    print(f"Signal found at {frame.index[i]}: {sig}")
        print(f"Total manual signals: {signals}")

except Exception as e:
    import traceback
    traceback.print_exc()

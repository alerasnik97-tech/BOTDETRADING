from pathlib import Path
import pandas as pd
from research_lab.data_loader import load_backtest_data_bundle
from research_lab.config import DEFAULT_PAIR, DEFAULT_DATA_DIRS, EngineConfig, NewsConfig, with_execution_mode
from research_lab.strategies.adr_exhaustion_fade import signal, NAME
from research_lab.strategies.common import is_in_session

def debug_spec():
    bundle = load_backtest_data_bundle(
        DEFAULT_PAIR,
        [Path(d) for d in DEFAULT_DATA_DIRS],
        "2020-01-01",
        "2025-12-31",
        "conservative_mode"
    )
    df = bundle.frame
    params = {"target_rr": 1.5, "break_even_at_r": None, "exhaustion_mult": 1.5, "session_name": "light_fixed"}
    
    signals = []
    total = len(df)
    print(f"Looping through {total} bars...")
    for i in range(1, total):
        if i % 5000 == 0:
            print(f"  Processed {i}/{total} bars...")
        sig = signal(df, i, params)
        if sig:
            signals.append((df.index[i], sig))
            
    print(f"Strategy: {NAME}")
    print(f"Total Signals Found: {len(signals)}")
    if signals:
        print("First 5 signals:")
        for t, s in signals[:5]:
            print(f"  {t}: {s}")

if __name__ == "__main__":
    debug_spec()

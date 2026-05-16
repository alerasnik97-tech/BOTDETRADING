import pandas as pd
import sys
import numpy as np
from pathlib import Path

# Agregar el directorio actual al path para importar research_lab
sys.path.append(".")

from research_lab.data_loader import load_backtest_data_bundle
from research_lab.config import DEFAULT_DATA_DIRS

try:
    data = load_backtest_data_bundle(
        pair="EURUSD",
        data_dirs=DEFAULT_DATA_DIRS,
        start="2024-01-01", # Solo un año para rapidez
        end="2024-02-01",
        execution_mode="conservative_mode",
        target_timeframe="M15"
    )
    frame = data.frame
    print(f"Dataframe rows: {len(frame)}")
    
    # Check specific columns
    cols = ["prev_day_high", "day_running_high", "high", "close"]
    sample = frame[cols].dropna().head(20)
    print("\nSample Data (First 20 non-NaN rows):")
    print(sample)
    
    # Check 11:00 NY specifically
    ny_11 = frame[frame.index.hour == 11].head(10)
    print("\nSample Data at 11:00 NY:")
    print(ny_11[cols])
    
    # Check why signals might be failing
    # Prev Day Sweep Reversion Logic:
    sweep_dist = 0.0 * 0.0001
    pdh = frame["prev_day_high"]
    day_h = frame["day_running_high"]
    close = frame["close"]
    
    potential_sweeps = (day_h > pdh + sweep_dist) & (frame.index.hour >= 11) & (frame.index.hour <= 15)
    print(f"\nPotential Sweeps found in this period: {potential_sweeps.sum()}")
    
    if potential_sweeps.any():
        print("\nExamples of potential sweeps:")
        print(frame.loc[potential_sweeps, ["prev_day_high", "day_running_high", "high", "close"]].head(5))

except Exception as e:
    import traceback
    traceback.print_exc()

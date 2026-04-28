
import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector
import pickle

def precompute():
    print("Precomputing signals (Debug)...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    period = "period_2020_2026"
    
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    sweeps = H1FractalSweepDetector({}).detect_sweeps(df_h1)
    print(f"Total sweeps: {len(sweeps)}")
    
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    
    choch_detector = First3MChochDetector({'max_mins_post_sweep': 60, 'sl_buffer': 0.5})
    # I'll manually run the loop to see progress
    signals = []
    df_m3_indexed = df_m3.set_index('timestamp_ny').sort_index()
    
    count = 0
    for sweep_time, direction in sweeps.items():
        count += 1
        if count % 500 == 0:
            print(f"Processing sweep {count}/{len(sweeps)}...")
            
        # Mocking First3MChochDetector.detect_choch logic for single sweep
        try:
            future = df_m3_indexed.loc[sweep_time:].iloc[0:21] # 60 mins = 20 bars
            # Simplified logic for debug
            signals.append({"choch_time": sweep_time, "direction": direction, "entry_price": 1.0, "sl_price": 0.9})
        except:
            continue
            
    print(f"Done. Detected {len(signals)} signals.")

if __name__ == "__main__":
    precompute()

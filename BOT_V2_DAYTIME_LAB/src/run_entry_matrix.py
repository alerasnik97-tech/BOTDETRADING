
import pandas as pd
import numpy as np
import json
from research_v2_engine import ResearchV2Engine, calculate_metrics
from pathlib import Path

def detect_choch(df, timestamp, direction):
    # Simplified CHoCH: break of last 3-candle fractal
    # This needs to be done efficiently
    idx = df.index.get_loc(timestamp)
    lookback = 10
    subset = df.iloc[max(0, idx-lookback):idx]
    
    if direction == 'SELL':
        # Bearish CHoCH: Break of previous fractal low
        # Fractal low: candle i with low < low(i-1) and low < low(i+1)
        # For simplicity in research: last significant low in lookback
        prev_low = subset['low'].min()
        if df.iloc[idx]['close'] < prev_low:
            return True
    else:
        # Bullish CHoCH: Break of previous fractal high
        prev_high = subset['high'].max()
        if df.iloc[idx]['close'] > prev_high:
            return True
    return False

def run_phase4_entries():
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    engine = ResearchV2Engine(manifest_path)
    base_level = "pdh" # PDH was best for shorts
    base_tf = "m3" # Assuming M3 is winner from Phase 3 based on manual edge
    
    # Load data
    print(f"Loading {base_tf} for Entry Matrix...")
    df_m1 = pd.concat([engine.load_prices('period_2015_2019', 'm1'), engine.load_prices('period_2020_2026', 'm1')]).sort_values('timestamp')
    df_m1.set_index('timestamp', inplace=True)
    df_tf = df_m1.resample('3min', closed='left', label='right').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).shift(1).dropna()
    
    df_h1 = pd.concat([engine.load_prices('period_2015_2019', 'h1'), engine.load_prices('period_2020_2026', 'h1')]).sort_values('timestamp')
    levels = engine.get_levels(df_h1)
    
    results = []
    
    # Entry Model 1: Simple Reclaim (Already in engine)
    config_reclaim = {'level_type': base_level, 'start_time': '08:30', 'end_time': '11:00', 'tp_r': 2.0, 'sl_value': 1.5, 'max_trades_per_day': 1}
    trades_reclaim = engine.run_simulation(df_tf, levels, config_reclaim)
    m_reclaim = calculate_metrics(trades_reclaim)
    m_reclaim['entry_model'] = 'Simple_Reclaim'
    results.append(m_reclaim)
    
    # Entry Model 2: CHoCH (Will require custom simulation loop or engine update)
    # For now, I'll report the simple reclaim as the baseline.
    
    df_results = pd.DataFrame(results)
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\entry_matrix")
    df_results.to_csv(out_dir / "entry_matrix_results.csv", index=False)
    print("Entry matrix analysis complete.")

if __name__ == "__main__":
    run_phase4_entries()



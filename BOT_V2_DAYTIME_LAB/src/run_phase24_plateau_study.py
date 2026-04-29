
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import pytz

# Add src to path
sys.path.append(str(Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\src")))
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector

def run_plateau_study():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB")
    output_dir = root / "outputs" / "phase24_controlled_optimization_2015_2026" / "plateau_search"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    manifest_path = root / "data" / "certified_m3" / "M3_CERTIFICATION_METADATA.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print("Loading M3 Data...")
    df_bid = pd.read_csv(manifest['bid_path'])
    df_ask = pd.read_csv(manifest['ask_path'])
    df_bid['timestamp'] = pd.to_datetime(df_bid['timestamp'], utc=True)
    df_ask['timestamp'] = pd.to_datetime(df_ask['timestamp'], utc=True)
    df_m3 = pd.merge(df_bid, df_ask, on='timestamp', suffixes=('_bid', '_ask'))
    tz_ny = pytz.timezone("America/New_York")
    df_m3['timestamp_ny'] = df_m3['timestamp'].dt.tz_convert(tz_ny)
    
    # News
    news_path = root / "data" / "news" / "news_events_2020_2026.csv"
    if news_path.exists():
        news_df = pd.read_csv(news_path)
        news_df['timestamp'] = pd.to_datetime(news_df['timestamp_utc'], utc=True)
    else:
        print("Warning: News file not found.")
        news_df = pd.DataFrame(columns=['timestamp'])

    # 2. Prep H1 for Sweeps
    df_m3.set_index('timestamp', inplace=True)
    df_h1 = df_m3.resample('1h').agg({
        'open_bid': 'first', 'high_bid': 'max', 'low_bid': 'min', 'close_bid': 'last',
        'timestamp_ny': 'first'
    }).dropna().reset_index()
    df_m3.reset_index(inplace=True)
    
    print(f"Detecting H1 Sweeps...")
    sweep_detector = H1FractalSweepDetector(params={})
    sweeps = sweep_detector.detect_sweeps(df_h1)
    
    # OPTIMIZATION: Filter sweeps to session +/- buffer
    sweeps['hour'] = sweeps['timestamp_ny'].dt.hour
    sweeps = sweeps[(sweeps['hour'] >= 6) & (sweeps['hour'] <= 16)]
    print(f"Sweeps filtered to session: {len(sweeps)}")
    
    # 3. Signal Search
    choch_detector = First3MChochDetector(params={'sl_buffer': 0.5, 'max_mins_post_sweep': 60})
    signals_base = choch_detector.detect_choch(df_m3, sweeps)
    print(f"Signals detected: {len(signals_base)}")
    
    # Pre-process signals
    df_m3_indexed = df_m3.set_index('timestamp_ny')
    signals_list = []
    for _, row in signals_base.iterrows():
        if row['choch_time'] in df_m3_indexed.index:
            idx_obj = df_m3_indexed.index.get_loc(row['choch_time'])
            if isinstance(idx_obj, slice): idx = idx_obj.start
            elif isinstance(idx_obj, np.ndarray): idx = idx_obj[0]
            else: idx = idx_obj
                
            signals_list.append({
                'index': idx, 'type': row['direction'], 'sl_custom': row['sl_price']
            })
    
    # 4. Grid Search
    engine = Phase14Engine(data_manifest_path=manifest_path)
    
    tp_range = [0.9, 1.1, 1.3, 1.5]
    be_range = [0.4, 0.5, 0.6, 1.0, None]
    
    results = []
    
    for tp in tp_range:
        for be in be_range:
            config = {
                "tp_r": tp, "be_r": be,
                "start_time": "07:00", "end_time": "16:30", "mandatory_close_time": "20:00",
                "max_trades_per_day": 1, "sl_buffer_pips": 0.5, "news_guard_mins": 30
            }
            trades_df = engine.run_backtest(df_m3, signals_list, news_df, config)
            metrics = engine.calculate_metrics(trades_df, config)
            metrics.update({"tp": tp, "be": be})
            results.append(metrics)
            print(f"Tested: TP={tp} BE={be} | PF: {metrics['pf']} | Sample: {metrics['sample']}")
            
    pd.DataFrame(results).to_csv(output_dir / "phase24_plateau_results.csv", index=False)
    print("Study completed.")

if __name__ == "__main__":
    run_plateau_study()

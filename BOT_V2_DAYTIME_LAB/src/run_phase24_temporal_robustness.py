
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

def run_temporal_robustness():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB")
    output_dir = root / "outputs" / "phase24_controlled_optimization_2015_2026" / "temporal_robustness"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    manifest_path = root / "data" / "certified_m3" / "M3_CERTIFICATION_METADATA.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    df_bid = pd.read_csv(manifest['bid_path'])
    df_ask = pd.read_csv(manifest['ask_path'])
    df_bid['timestamp'] = pd.to_datetime(df_bid['timestamp'], utc=True)
    df_ask['timestamp'] = pd.to_datetime(df_ask['timestamp'], utc=True)
    df_m3 = pd.merge(df_bid, df_ask, on='timestamp', suffixes=('_bid', '_ask'))
    tz_ny = pytz.timezone("America/New_York")
    df_m3['timestamp_ny'] = df_m3['timestamp'].dt.tz_convert(tz_ny)
    
    # News
    news_path = root / "data" / "news" / "news_events_2020_2026.csv"
    news_df = pd.read_csv(news_path) if news_path.exists() else pd.DataFrame(columns=['timestamp'])
    if not news_df.empty: news_df['timestamp'] = pd.to_datetime(news_df['timestamp_utc'], utc=True)

    # 2. Signals
    df_m3.set_index('timestamp', inplace=True)
    df_h1 = df_m3.resample('1h').agg({'open_bid': 'first', 'high_bid': 'max', 'low_bid': 'min', 'close_bid': 'last', 'timestamp_ny': 'first'}).dropna().reset_index()
    df_m3.reset_index(inplace=True)
    
    sweeps = H1FractalSweepDetector(params={}).detect_sweeps(df_h1)
    sweeps['hour'] = sweeps['timestamp_ny'].dt.hour
    sweeps = sweeps[(sweeps['hour'] >= 6) & (sweeps['hour'] <= 16)]
    
    signals_base = First3MChochDetector(params={'sl_buffer': 0.5, 'max_mins_post_sweep': 60}).detect_choch(df_m3, sweeps)
    
    df_m3_indexed = df_m3.set_index('timestamp_ny')
    signals_list = []
    for _, row in signals_base.iterrows():
        if row['choch_time'] in df_m3_indexed.index:
            idx_obj = df_m3_indexed.index.get_loc(row['choch_time'])
            idx = idx_obj.start if isinstance(idx_obj, slice) else (idx_obj[0] if isinstance(idx_obj, np.ndarray) else idx_obj)
            signals_list.append({'index': idx, 'type': row['direction'], 'sl_custom': row['sl_price']})
    
    # 3. Candidates to Test
    candidates = [
        {"name": "Phase22_Base", "tp": 1.1, "be": 0.5},
        {"name": "Phase24_Robust_Peak", "tp": 1.3, "be": 0.5},
        {"name": "Phase24_High_PF", "tp": 1.5, "be": 0.4}
    ]
    
    engine = Phase14Engine(data_manifest_path=manifest_path)
    yearly_results = []
    
    df_m3['year'] = df_m3['timestamp_ny'].dt.year
    years = sorted(df_m3['year'].unique())
    
    for cand in candidates:
        config = {
            "tp_r": cand['tp'], "be_r": cand['be'],
            "start_time": "07:00", "end_time": "16:30", "mandatory_close_time": "20:00",
            "max_trades_per_day": 1, "sl_buffer_pips": 0.5, "news_guard_mins": 30
        }
        
        for year in years:
            df_year = df_m3[df_m3['year'] == year].copy()
            # Filter signals for this year
            year_start = df_year['timestamp_ny'].min()
            year_end = df_year['timestamp_ny'].max()
            signals_year = [s for s in signals_list if year_start <= df_m3.iloc[s['index']]['timestamp_ny'] <= year_end]
            
            # Adjust indexes to be local to df_year or pass full df with year filter?
            # Easiest: run full backtest and filter results by year.
            pass
            
        full_trades = engine.run_backtest(df_m3, signals_list, news_df, config)
        full_trades['year'] = full_trades['entry_time'].dt.year
        
        for year in years:
            yt = full_trades[full_trades['year'] == year]
            m = engine.calculate_metrics(yt, config)
            m.update({"strategy": cand['name'], "year": year})
            yearly_results.append(m)
            
    pd.DataFrame(yearly_results).to_csv(output_dir / "phase24_robustness_by_year.csv", index=False)
    print("Robustness study completed.")

if __name__ == "__main__":
    run_temporal_robustness()

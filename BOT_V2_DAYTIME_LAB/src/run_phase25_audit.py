
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

def run_phase25_audit():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB")
    output_dir = root / "outputs" / "phase25_max_robust_plateau"
    os.makedirs(output_dir / "plateau_search", exist_ok=True)
    os.makedirs(output_dir / "regression_boundary", exist_ok=True)
    
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
    news_df = pd.read_csv(news_path) if news_path.exists() else pd.DataFrame(columns=['timestamp'])
    if not news_df.empty: news_df['timestamp'] = pd.to_datetime(news_df['timestamp_utc'], utc=True)

    # 2. Prep Signals
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
    
    print(f"Signals detected: {len(signals_list)}")

    # 3. Grid Audit
    engine = Phase14Engine(data_manifest_path=manifest_path)
    results = []
    
    tp_range = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    be_range = [0.4, 0.5, 0.6, 0.75, 1.0, None]
    
    for tp in tp_range:
        for be in be_range:
            config = {
                "tp_r": tp, "be_r": be,
                "start_time": "07:00", "end_time": "16:30", "mandatory_close_time": "20:00",
                "max_trades_per_day": 1, "sl_buffer_pips": 0.5, "news_guard_mins": 30
            }
            trades_df = engine.run_backtest(df_m3, signals_list, news_df, config)
            
            if not trades_df.empty:
                # Calc r_return manually
                def calc_r(row):
                    dist = (row['exit_price'] - row['entry_price']) if row['type'] == 'LONG' else (row['entry_price'] - row['exit_price'])
                    return dist / row['risk']
                trades_df['r_return'] = trades_df.apply(calc_r, axis=1)
                
                # PF and Expectancy
                profits = trades_df[trades_df['r_return'] > 0]['r_return'].sum()
                losses = abs(trades_df[trades_df['r_return'] < 0]['r_return'].sum())
                pf = round(profits / losses if losses > 0 else profits, 2)
                exp = round(trades_df['r_return'].mean(), 3)
                
                # DD
                trades_df['cum_pnl'] = trades_df['r_return'].cumsum()
                max_dd = (trades_df['cum_pnl'] - trades_df['cum_pnl'].cummax()).min()
            else:
                pf, exp, max_dd = 0, 0, 0
            
            metrics = {"tp": tp, "be": be, "pf": pf, "exp": exp, "max_dd": round(max_dd, 2), "sample": len(trades_df)}
            results.append(metrics)
            print(f"Audit: TP={tp} BE={be} | PF: {pf} | DD: {max_dd:.2f}")
            
    pd.DataFrame(results).to_csv(output_dir / "plateau_search" / "phase25_plateau_audit.csv", index=False)
    print("Audit completed.")

if __name__ == "__main__":
    run_phase25_audit()

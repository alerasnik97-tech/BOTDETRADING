
import pandas as pd
import numpy as np
import json
from pathlib import Path
from phase6_engine import Phase6Engine
from datetime import datetime
import os
import sys

def run_baseline():
    print("Calculating Phase 7 Baseline (M3 First CHoCH)...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    engine = Phase6Engine()
    periods = ['period_2015_2019', 'period_2020_2026']
    
    # Common news
    news_list = []
    for p in periods:
        if 'news' in manifest[p]:
            news_list.append(pd.read_csv(manifest[p]['news']))
    news_df = pd.concat(news_list)
    
    tf = 'm3'
    config = {
        'entry_type': 1, 'timeframe': tf, 'fractal_n': 3,
        'start_hour': '08:30', 'end_hour': '11:00',
        'tp_val': 2.0, 'be_r': None, 'sl_type': 'sweep',
        'sl_plus_pips': 0.5, 'news_block_mins': 30,
        'one_trade_per_day': True
    }
    
    output_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase7_m3_choch_refinement\baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_trades_list = []
    
    for p in periods:
        print(f"  Processing {p}...", flush=True)
        # Load M3 data (resample if needed)
        source_key = 'm5_bid' if 'm5_bid' in manifest[p] else 'm1_bid'
        print(f"    Reading {source_key}...", flush=True)
        df_src = pd.read_csv(manifest[p][source_key])
        print(f"    Converting timestamps...", flush=True)
        df_src['timestamp'] = pd.to_datetime(df_src['timestamp'], utc=True)
        df_src.set_index('timestamp', inplace=True)
        print(f"    Resampling to 3min...", flush=True)
        df = df_src.resample('3min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
        
        print(f"    Calculating fractals (N={config['fractal_n']})...", flush=True)
        df['timestamp_ny'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        df['is_high_fractal'], df['is_low_fractal'] = engine.get_fractals(df, n=config['fractal_n'])
        
        # Load H1 levels
        print(f"    Loading H1 levels...", flush=True)
        df_h1 = pd.read_csv(manifest[p]['h1_bid'])
        df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        levels = engine.get_levels(df_h1)
        
        print(f"    Starting backtest for {p}...", flush=True)
        trades = engine.run_phase6_backtest(df, levels, news_df, config)
        if not trades.empty:
            all_trades_list.append(trades)
    
    if not all_trades_list:
        print("FAIL: No trades found.")
        return

    full_trades = pd.concat(all_trades_list).sort_values('entry_time')
    full_trades.to_csv(output_dir / "baseline_phase7_trades.csv", index=False)
    
    # Calculate Metrics
    total = len(full_trades)
    gp = full_trades[full_trades['r_value'] > 0]['r_value'].sum()
    gl = abs(full_trades[full_trades['r_value'] < 0]['r_value'].sum())
    pf = gp / gl if gl > 0 else 0
    wr = len(full_trades[full_trades['result'] == 'TP']) / total if total > 0 else 0
    
    summary = {
        'sample': total,
        'pf': round(pf, 2),
        'expectancy_R': round(full_trades['r_value'].mean(), 3),
        'win_rate': round(wr, 4),
        'total_r': round(full_trades['r_value'].sum(), 2)
    }
    
    with open(output_dir / "baseline_phase7_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Baseline Complete: PF={pf:.2f}, Sample={total}")

if __name__ == "__main__":
    run_baseline()



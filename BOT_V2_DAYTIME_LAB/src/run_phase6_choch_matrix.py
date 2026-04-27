
import pandas as pd
import numpy as np
import json
from pathlib import Path
from phase6_engine import Phase6Engine
from datetime import datetime
import os
import sys

def run_matrix():
    print("Starting Phase 6 Matrix Research...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    engine = Phase6Engine()
    periods = ['period_2020_2026']
    
    # Common news
    news_list = []
    for p in periods:
        if 'news' in manifest[p]:
            news_list.append(pd.read_csv(manifest[p]['news']))
    news_df = pd.concat(news_list)
    
    # Test Parameters
    timeframes = ['m15', 'm5']
    entry_types = [1] 
    tp_vals = [2.0, 3.0]
    sl_types = ['sweep']
    fractal_ns = [3]
    results = []
    output_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase6_choch_entries")
    output_dir.mkdir(parents=True, exist_ok=True)

    for tf in timeframes:
        print(f"--- Timeframe: {tf} ---")
        sys.stdout.flush()
        
        # Pre-load all periods for this TF
        tf_period_data = []
        for p in periods:
            key = f'{tf}_bid'
            if key in manifest[p]:
                df = pd.read_csv(manifest[p][key])
            else:
                source_key = 'm5_bid' if 'm5_bid' in manifest[p] else 'm1_bid'
                df_src = pd.read_csv(manifest[p][source_key])
                df_src['timestamp'] = pd.to_datetime(df_src['timestamp'], utc=True)
                df_src.set_index('timestamp', inplace=True)
                tf_map = {'m15': '15min', 'm3': '3min', 'm5': '5min'}
                df = df_src.resample(tf_map.get(tf, tf)).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
            
            df['timestamp_ny'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
            
            # Pre-calculate fractals (N=3)
            df['is_high_fractal'], df['is_low_fractal'] = engine.get_fractals(df, n=3)
            
            # Load H1 levels
            df_h1 = pd.read_csv(manifest[p]['h1_bid'])
            df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
            levels = engine.get_levels(df_h1)
            
            tf_period_data.append((df, levels))

        # Matrix loop inside Timeframe
        for et in entry_types:
            for sl_t in sl_types:
                for tp in tp_vals:
                    config = {
                        'entry_type': et, 'timeframe': tf, 'fractal_n': 3,
                        'start_hour': '08:00', 'end_hour': '12:00',
                        'tp_val': tp, 'be_r': 1.0, 'sl_type': sl_t,
                        'sl_plus_pips': 0.5, 'news_block_mins': 30
                    }
                    
                    config_trades = []
                    for df_p, levels_p in tf_period_data:
                        # Modified engine call that skips fractal calculation
                        trades = engine.run_phase6_backtest(df_p, levels_p, news_df, config)
                        config_trades.append(trades)
                    
                    full_trades = pd.concat(config_trades)
                    if not full_trades.empty:
                        gp = full_trades[full_trades['r_value'] > 0]['r_value'].sum()
                        gl = abs(full_trades[full_trades['r_value'] < 0]['r_value'].sum())
                        pf = gp / gl if gl > 0 else 0
                        
                        res = {
                            'tf': tf, 'entry_type': et, 'sl_type': sl_t, 'tp_val': tp,
                            'sample': len(full_trades), 'pf': round(pf, 2),
                            'expectancy': round(full_trades['r_value'].mean(), 3)
                        }
                        results.append(res)
                        print(f"  ET{et} SL{sl_t} TP{tp}: PF={pf:.2f} Sample={len(full_trades)}")
                        sys.stdout.flush()
                        
                        if pf >= 1.30:
                            full_trades.to_csv(output_dir / f"trades_tf{tf}_et{et}_sl{sl_t}_tp{tp}.csv", index=False)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / "phase6_entry_family_results.csv", index=False)
    print("Matrix execution complete.")

if __name__ == "__main__":
    run_matrix()



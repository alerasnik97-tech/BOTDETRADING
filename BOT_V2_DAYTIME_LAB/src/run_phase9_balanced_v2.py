
import pandas as pd
import numpy as np
import json
from pathlib import Path
from phase6_engine import Phase6Engine

def run_balanced_v2():
    print("Phase 6: Balanced_V2 Frequency Test")
    engine = Phase6Engine()
    
    # Base Config: High Precision but relaxed for frequency
    config = {
        'entry_type': 1, 'timeframe': 'm3', 'fractal_n': 8,
        'start_hour': '08:00', 'end_hour': '13:00', # Expanded window
        'tp_val': 1.5, 'be_r': None, 'sl_type': 'sweep',
        'sl_plus_pips': 0.5, 'news_block_mins': 30,
        'one_trade_per_day': False, # Allow multiple trades
        'first_sweep_only': True,   # BUT ONLY FIRST SWEEP PER LEVEL (Engine Logic)
        'min_atr': 0.0010, 'trend_exhaustion': True,
        'min_body_pct': 0.40,        # Relaxed body
        'exclude_friday': False      # Include fridays
    }
    
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    periods = ['period_2015_2019', 'period_2020_2026']
    news_df = pd.concat([pd.read_csv(manifest[p]['news']) for p in periods])
    
    all_trades = []
    for p in periods:
        df_src = pd.read_csv(manifest[p]['m5_bid'])
        df_src['timestamp'] = pd.to_datetime(df_src['timestamp'], utc=True)
        df_src.set_index('timestamp', inplace=True)
        df_m3 = df_src.resample('3min', closed='left', label='right').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).shift(1).dropna().reset_index()
        df_m3['timestamp_ny'] = pd.to_datetime(df_m3['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        df_m3['is_high_fractal'], df_m3['is_low_fractal'] = engine.get_fractals(df_m3, n=8)
        df_h1 = pd.read_csv(manifest[p]['h1_bid'])
        df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        levels = engine.get_levels(df_h1)
        trades = engine.run_phase6_backtest(df_m3, levels, news_df, config)
        all_trades.append(trades)
    
    full_trades = pd.concat(all_trades)
    total_months = 132
    
    gp = full_trades[full_trades['r_value'] > 0]['r_value'].sum()
    gl = abs(full_trades[full_trades['r_value'] < 0]['r_value'].sum())
    pf = gp / gl if gl > 0 else 0
    
    print(f"Balanced_V2: PF={pf:.3f} Sample={len(full_trades)} Freq={len(full_trades)/total_months:.2f} trades/mes")
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase9_frequency_expansion\final_frequency_combinations")
    full_trades.to_csv(out_dir / "Balanced_V2_trades.csv", index=False)

if __name__ == "__main__":
    run_balanced_v2()



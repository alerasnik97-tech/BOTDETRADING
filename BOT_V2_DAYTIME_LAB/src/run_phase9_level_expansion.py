
import pandas as pd
import numpy as np
import json
from pathlib import Path
from phase6_engine import Phase6Engine

def run_level_expansion():
    print("Phase 2: Level Expansion Laboratory")
    engine = Phase6Engine()
    
    # Base Config (Candidate B rules)
    config = {
        'entry_type': 1, 'timeframe': 'm3', 'fractal_n': 8,
        'start_hour': '08:30', 'end_hour': '11:00',
        'tp_val': 1.5, 'be_r': None, 'sl_type': 'sweep',
        'sl_plus_pips': 0.5, 'news_block_mins': 30,
        'one_trade_per_day': True, 'first_sweep_only': True,
        'min_atr': 0.0012, 'trend_exhaustion': True,
        'min_body_pct': 0.60
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
        
        # This will now include Asia/London because of the engine update
        trades = engine.run_phase6_backtest(df_m3, levels, news_df, config)
        all_trades.append(trades)
    
    full_trades = pd.concat(all_trades)
    
    # Analyze by level type
    level_types = full_trades.groupby('level')['r_value'].agg(['count', 'sum'])
    level_types['pf'] = full_trades.groupby('level').apply(lambda x: x[x['r_value']>0]['r_value'].sum() / abs(x[x['r_value']<0]['r_value'].sum()) if any(x['r_value']<0) else 1.0)
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase9_frequency_expansion\individual_relaxations")
    out_dir.mkdir(parents=True, exist_ok=True)
    level_types.to_csv(out_dir / "level_expansion_results.csv")
    
    print("Level Expansion Laboratory Complete.")
    print(level_types)

if __name__ == "__main__":
    run_level_expansion()



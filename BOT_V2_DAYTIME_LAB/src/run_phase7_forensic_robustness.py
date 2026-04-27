
import pandas as pd
import numpy as np
import json
from pathlib import Path
from phase6_engine import Phase6Engine

def run_robustness_audit():
    print("Phase 6: Robustness Audit - STRONG_CANDIDATE_PHASE7_V1")
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    
    engine = Phase6Engine()
    periods = ['period_2015_2019', 'period_2020_2026']
    news_df = pd.concat([pd.read_csv(manifest[p]['news']) for p in periods if 'news' in manifest[p]])
    
    config = {
        'entry_type': 1, 'timeframe': 'm3', 'fractal_n': 8,
        'start_hour': '08:30', 'end_hour': '11:00',
        'tp_val': 1.5, 'be_r': None, 'sl_type': 'sweep',
        'sl_plus_pips': 0.5, 'news_block_mins': 30,
        'one_trade_per_day': True, 'first_sweep_only': True,
        'min_atr': 0.0012, 'trend_exhaustion': True
    }
    
    all_trades = []
    for p in periods:
        df_src = pd.read_csv(manifest[p]['m5_bid'])
        df_src['timestamp'] = pd.to_datetime(df_src['timestamp'], utc=True)
        df_src.set_index('timestamp', inplace=True)
        df_m3 = df_src.resample('3min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
        df_m3['timestamp_ny'] = pd.to_datetime(df_m3['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        df_m3['is_high_fractal'], df_m3['is_low_fractal'] = engine.get_fractals(df_m3, n=8)
        df_h1 = pd.read_csv(manifest[p]['h1_bid'])
        df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        levels = engine.get_levels(df_h1)
        trades = engine.run_phase6_backtest(df_m3, levels, news_df, config)
        all_trades.append(trades)
    
    full_trades = pd.concat(all_trades)
    full_trades['entry_time'] = pd.to_datetime(full_trades['entry_time'])
    full_trades['year'] = full_trades['entry_time'].dt.year
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase7_forensic_audit\robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Robustness by Year
    yearly = []
    for yr in sorted(full_trades['year'].unique()):
        df_yr = full_trades[full_trades['year'] == yr]
        gp = df_yr[df_yr['r_value'] > 0]['r_value'].sum()
        gl = abs(df_yr[df_yr['r_value'] < 0]['r_value'].sum())
        pf = gp / gl if gl > 0 else (99 if gp > 0 else 0)
        yearly.append({
            'year': yr, 'sample': len(df_yr), 'pf': round(pf, 2), 
            'expectancy': round(df_yr['r_value'].mean(), 3),
            'cumulative_r': round(df_yr['r_value'].sum(), 2)
        })
    pd.DataFrame(yearly).to_csv(out_dir / "robustness_by_year.csv", index=False)
    
    # Robustness by Period (Blocks)
    blocks = [
        (2015, 2017), (2018, 2019), (2020, 2022), (2023, 2026)
    ]
    block_res = []
    for s, e in blocks:
        df_b = full_trades[(full_trades['year'] >= s) & (full_trades['year'] <= e)]
        gp = df_b[df_b['r_value'] > 0]['r_value'].sum()
        gl = abs(df_b[df_b['r_value'] < 0]['r_value'].sum())
        pf = gp / gl if gl > 0 else 0
        block_res.append({
            'period': f"{s}-{e}", 'sample': len(df_b), 'pf': round(pf, 2),
            'cumulative_r': round(df_b['r_value'].sum(), 2)
        })
    pd.DataFrame(block_res).to_csv(out_dir / "robustness_by_period.csv", index=False)
    print("Robustness Audit Complete.")

if __name__ == "__main__":
    run_robustness_audit()



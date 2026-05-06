
import pandas as pd
import numpy as np
import json
from pathlib import Path
from phase6_engine import Phase6Engine

def run_phase9_combinations():
    print("Phase 6: Frequency Combinations Laboratory")
    engine = Phase6Engine()
    
    combinations = [
        {
            "name": "Balanced_Candidate_M5",
            "config": {
                'start_hour': '08:30', 'end_hour': '12:00',
                'min_body_pct': 0.50, 'one_trade_per_day': False, 'first_sweep_only': False,
                'exclude_friday': False
            }
        },
        {
            "name": "High_Freq_Candidate_M10",
            "config": {
                'start_hour': '08:00', 'end_hour': '14:00',
                'min_body_pct': 0.40, 'one_trade_per_day': False, 'first_sweep_only': False,
                'exclude_friday': False, 'min_atr': 0.0010
            }
        }
    ]
    
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    periods = ['period_2015_2019', 'period_2020_2026']
    news_df = pd.concat([pd.read_csv(manifest[p]['news']) for p in periods])
    
    results = []
    for comb in combinations:
        print(f"  Testing {comb['name']}...")
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
            
            trades = engine.run_phase6_backtest(df_m3, levels, news_df, comb['config'])
            
            # Post-hoc filters if any
            if comb['config'].get('exclude_friday'):
                trades['weekday'] = pd.to_datetime(trades['entry_time'], utc=True).dt.day_name()
                trades = trades[trades['weekday'] != "Friday"]
                
            all_trades.append(trades)
        
        full_trades = pd.concat(all_trades)
        total_months = 132
        
        gp = full_trades[full_trades['r_value'] > 0]['r_value'].sum()
        gl = abs(full_trades[full_trades['r_value'] < 0]['r_value'].sum())
        pf = gp / gl if gl > 0 else 0
        
        # Output trades for robustness
        out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase9_frequency_expansion\final_frequency_combinations")
        out_dir.mkdir(parents=True, exist_ok=True)
        full_trades.to_csv(out_dir / f"{comb['name']}_trades.csv", index=False)
        
        results.append({
            "name": comb['name'],
            "sample": len(full_trades),
            "trades_per_month": round(len(full_trades)/total_months, 2),
            "pf": round(pf, 3),
            "expectancy": round(full_trades['r_value'].mean(), 4)
        })
        
    pd.DataFrame(results).to_csv(out_dir / "phase9_final_combinations.csv", index=False)
    print("Combinations Complete.")

if __name__ == "__main__":
    run_phase9_combinations()



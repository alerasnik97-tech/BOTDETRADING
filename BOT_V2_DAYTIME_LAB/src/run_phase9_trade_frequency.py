
import pandas as pd
import numpy as np
import json
from pathlib import Path
from phase6_engine import Phase6Engine

def run_frequency_test(name, config_patch):
    print(f"  Testing Frequency Variant: {name}...")
    engine = Phase6Engine()
    
    # Base Config (Phase 8 Candidate B rules)
    config = {
        'entry_type': 1, 'timeframe': 'm3', 'fractal_n': 8,
        'start_hour': '08:30', 'end_hour': '11:00',
        'tp_val': 1.5, 'be_r': None, 'sl_type': 'sweep',
        'sl_plus_pips': 0.5, 'news_block_mins': 30,
        'one_trade_per_day': True, 'first_sweep_only': True,
        'min_atr': 0.0012, 'trend_exhaustion': True,
        'min_body_pct': 0.60
    }
    config.update(config_patch)
    
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    periods = ['period_2015_2019', 'period_2020_2026']
    news_df = pd.concat([pd.read_csv(manifest[p]['news']) for p in periods])
    
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
    total_months = 132
    
    gp = full_trades[full_trades['r_value'] > 0]['r_value'].sum()
    gl = abs(full_trades[full_trades['r_value'] < 0]['r_value'].sum())
    pf = gp / gl if gl > 0 else 0
    
    return {
        "variant": name,
        "sample": len(full_trades),
        "trades_per_month": round(len(full_trades)/total_months, 2),
        "pf": round(pf, 3),
        "expectancy": round(full_trades['r_value'].mean(), 4)
    }

def run_phase4():
    print("Phase 4: 1 vs 2 Trades Per Day Laboratory")
    results = []
    
    # 1. Baseline (1 Trade/Day)
    results.append(run_frequency_test("Baseline_1_Trade_Day", {"one_trade_per_day": True}))
    
    # 3. No Limit (Multiple trades/day)
    results.append(run_frequency_test("No_Limit_Trades_Day", {"one_trade_per_day": False}))
    
    # 4. No Limit Sweeps (Multiple sweeps/day)
    results.append(run_frequency_test("No_Limit_Sweeps", {"first_sweep_only": False}))
    
    # 5. Aggressive Freq (Expanded Window + No Limit)
    results.append(run_frequency_test("Aggressive_Freq_Window_No_Limit", {
        "one_trade_per_day": False, "start_hour": "08:00", "end_hour": "14:00", "first_sweep_only": False
    }))

    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase9_frequency_expansion\trade_frequency")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "one_vs_two_trades_results.csv", index=False)
    print("Frequency Laboratory Complete.")

if __name__ == "__main__":
    run_phase4()



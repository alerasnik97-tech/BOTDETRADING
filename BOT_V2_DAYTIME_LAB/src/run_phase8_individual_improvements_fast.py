
import pandas as pd
import numpy as np
import json
from pathlib import Path
from phase6_engine import Phase6Engine

def run_fast_improvements():
    print("Phase 2: Fast Individual Improvements Laboratory")
    
    # 1. Load Baseline Trades
    trades_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase8_improvement_lab\baseline_lock\phase7_repaired_baseline_trades.csv"
    if not Path(trades_path).exists():
        print("Baseline trades not found. Run reproduction first.")
        return
        
    trades = pd.read_csv(trades_path)
    trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True)
    trades['weekday'] = trades['entry_time'].dt.day_name()
    
    results = []
    
    def calc_metrics(df, name):
        gp = df[df['r_value'] > 0]['r_value'].sum()
        gl = abs(df[df['r_value'] < 0]['r_value'].sum())
        pf = gp / gl if gl > 0 else 0
        return {
            "variant": name,
            "sample": len(df),
            "pf": round(pf, 3),
            "expectancy": round(df['r_value'].mean(), 4),
            "cumulative_r": round(df['r_value'].sum(), 2)
        }

    # Variants (Fast Filtering)
    results.append(calc_metrics(trades, "Baseline"))
    results.append(calc_metrics(trades[trades['max_depth_pips'] >= 1.0], "Sweep_Min_1_Pip"))
    results.append(calc_metrics(trades[trades['max_depth_pips'] <= 15.0], "Sweep_Max_15_Pips"))
    results.append(calc_metrics(trades[trades['max_depth_pips'] <= 20.0], "Sweep_Max_20_Pips"))
    results.append(calc_metrics(trades[trades['body_pct'] >= 0.50], "CHoCH_Body_50"))
    results.append(calc_metrics(trades[trades['body_pct'] >= 0.60], "CHoCH_Body_60"))
    results.append(calc_metrics(trades[trades['weekday'] != "Friday"], "Exclude_Friday"))
    results.append(calc_metrics(trades[trades['level'] != "pdh"], "Exclude_PDH"))
    results.append(calc_metrics(trades[trades['time_post_sweep'] <= 60.0], "Sweep_to_Entry_LE_60m"))
    results.append(calc_metrics(trades[trades['time_post_sweep'] <= 45.0], "Sweep_to_Entry_LE_45m"))

    # Variants (Requires Re-run)
    print("  Running re-run variants (Management)...")
    engine = Phase6Engine()
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    periods = ['period_2015_2019', 'period_2020_2026']
    news_df = pd.concat([pd.read_csv(manifest[p]['news']) for p in periods])
    
    config_tp2 = {
        'entry_type': 1, 'timeframe': 'm3', 'fractal_n': 8,
        'start_hour': '08:30', 'end_hour': '11:00',
        'tp_val': 2.0, 'be_r': None, 'sl_type': 'sweep',
        'sl_plus_pips': 0.5, 'news_block_mins': 30,
        'one_trade_per_day': True, 'first_sweep_only': True,
        'min_atr': 0.0012, 'trend_exhaustion': True
    }
    
    tp2_trades = []
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
        tp2_trades.append(engine.run_phase6_backtest(df_m3, levels, news_df, config_tp2))
        
    results.append(calc_metrics(pd.concat(tp2_trades), "TP_2.0R"))

    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase8_improvement_lab\individual_improvements")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "individual_improvement_results.csv", index=False)
    print("Fast Improvements Complete.")

if __name__ == "__main__":
    run_fast_improvements()



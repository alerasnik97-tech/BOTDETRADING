
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from phase12_advanced_engine import Phase12AdvancedEngine

def run_robustness():
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\ARCHIVE_SUPERSEDED\duplicated_folders\Bot V1_PENDING_DELETE\data_manifest\certified_data_paths.json"
    engine = Phase12AdvancedEngine()
    
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    periods = ['period_2015_2019', 'period_2020_2026']
    
    news_list = []
    for p in periods:
        news_list.append(pd.read_csv(manifest[p]['news']))
    news_df = pd.concat(news_list)
    
    h1_list = []
    for p in periods:
        df = pd.read_csv(manifest[p]['h1_bid'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        h1_list.append(df)
    df_h1 = pd.concat(h1_list).sort_values('timestamp')
    levels = engine.get_levels(df_h1)
    
    ltf_list = []
    for p in periods:
        df_src = pd.read_csv(manifest[p]['m5_bid'])
        df_src['timestamp'] = pd.to_datetime(df_src['timestamp'], utc=True)
        df_src.set_index('timestamp', inplace=True)
        df_m3 = df_src.resample('3min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
        df_m3['timestamp_ny'] = df_m3['timestamp'].dt.tz_convert(engine.tz_ny)
        ltf_list.append(df_m3)
    df_ltf = pd.concat(ltf_list).sort_values('timestamp').reset_index(drop=True)
    
    # Candidate: Phase 7
    config_base = {
        "start_hour": "08:30",
        "end_hour": "11:00",
        "tp_r": 1.5,
        "be_r": None,
        "fractal_n": 3,
        "min_body_pct": 0.0,
        "sl_plus_pips": 0.5,
        "one_trade_per_day": True,
        "news_block_mins": 30
    }
    
    robustness_results = []
    
    # 1. Slippage Stress
    for slip in [0.0, 0.5, 1.0, 1.5]:
        print(f"Testing Slippage {slip} pips...")
        config = config_base.copy()
        config['sl_plus_pips'] = 0.5 + slip # Simulate worse SL
        # Note: advanced engine doesn't take slip for entry yet, 
        # but sl_plus_pips increases risk and worsens TP price.
        
        trades = engine.run_backtest(df_ltf, levels, news_df, config)
        if not trades.empty:
            gp = trades[trades['r_value'] > 0]['r_value'].sum()
            gl = abs(trades[trades['r_value'] < 0]['r_value'].sum())
            pf = gp / gl if gl > 0 else 0
            robustness_results.append({"test": "Slippage", "value": slip, "pf": round(pf, 2)})

    # 2. Window Shift
    for shift in [-30, 0, 30]:
        print(f"Testing Window Shift {shift} mins...")
        config = config_base.copy()
        start = (datetime.strptime(config['start_hour'], "%H:%M") + timedelta(minutes=shift)).strftime("%H:%M")
        end = (datetime.strptime(config['end_hour'], "%H:%M") + timedelta(minutes=shift)).strftime("%H:%M")
        config['start_hour'] = start
        config['end_hour'] = end
        
        trades = engine.run_backtest(df_ltf, levels, news_df, config)
        if not trades.empty:
            gp = trades[trades['r_value'] > 0]['r_value'].sum()
            gl = abs(trades[trades['r_value'] < 0]['r_value'].sum())
            pf = gp / gl if gl > 0 else 0
            robustness_results.append({"test": "WindowShift", "value": shift, "pf": round(pf, 2)})

    output_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase12_surpass_manual_pf\robustness")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(robustness_results).to_csv(output_dir / "phase7_robustness_test.csv", index=False)
    print("Robustness Complete.")

if __name__ == "__main__":
    run_robustness()

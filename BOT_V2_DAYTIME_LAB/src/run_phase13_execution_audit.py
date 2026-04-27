
import pandas as pd
import json
from pathlib import Path
from phase13_engine import Phase13Engine
from phase13_helpers import get_all_levels
import os
import numpy as np

def run_execution_audit():
    print("Starting Phase 13 Execution & Cost Audit...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase13Engine(manifest_path)
    
    period = "period_2020_2026"
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    news_df = engine.load_news(period)
    levels = get_all_levels(engine, df_m3, period)
    
    # Bias maps
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_h1['ema_20'] = df_h1['close_bid'].ewm(span=20, adjust=False).mean()
    df_h1['bias'] = np.where(df_h1['close_bid'] > df_h1['ema_20'], 1, -1)
    bias_map_h1 = df_h1.set_index('timestamp')['bias'].to_dict()

    df_m15 = engine.load_and_prep_prices(period, timeframe='m15')
    df_m15['ema_20'] = df_m15['close_bid'].ewm(span=20, adjust=False).mean()
    df_m15['bias'] = np.where(df_m15['close_bid'] > df_m15['ema_20'], 1, -1)
    df_m15.set_index('timestamp', inplace=True)
    m15_bias_series = df_m15['bias'].reindex(df_m3['timestamp'], method='ffill')
    bias_map_m15 = m15_bias_series.to_dict()
    
    results = []
    output_base = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase13_two_practical_entries\execution")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Stress tests
    spread_stresses = [0.0, 0.2, 0.5, 1.0] # in pips
    
    for stress in spread_stresses:
        print(f"Stressing Spread: +{stress} pips...")
        # To stress spread, we temporarily modify ask prices in a copy
        df_stressed = df_m3.copy()
        df_stressed['open_ask'] += stress * 0.0001
        df_stressed['high_ask'] += stress * 0.0001
        df_stressed['low_ask'] += stress * 0.0001
        df_stressed['close_ask'] += stress * 0.0001
        
        config = {
            "method": "reclaim", "params": {"reclaim_body_pct": 0.6, "session_type": "london", "use_bias": True, "bias_map": bias_map_h1},
            "tp_r": 2.5, "sl_pips_plus": 1.0, "news_guard_mins": 30, "start_time": "03:00", "end_time": "12:00",
            "one_trade_per_day": True, "use_m15_bias": True, "m15_bias_map": bias_map_m15
        }
        
        trades = engine.run_backtest(df_stressed, levels, news_df, config)
        metrics = engine.calculate_metrics(trades)
        metrics['spread_stress_pips'] = stress
        results.append(metrics)
        print(f"  +{stress} pips: PF={metrics['pf']}, Sample={metrics['sample']}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_base / "phase13_spread_stress_test.csv", index=False)
    print("Execution Audit Complete.")

if __name__ == "__main__":
    run_execution_audit()

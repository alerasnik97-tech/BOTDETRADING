
import pandas as pd
import json
from pathlib import Path
from phase13_engine import Phase13Engine
from phase13_helpers import get_all_levels
import os
import numpy as np

def run_screening():
    print("Starting Phase 13 Screening Pass 9 (Dual Bias H1+M15)...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase13Engine(manifest_path)
    
    period = "period_2020_2026"
    print(f"Loading M3 data for {period}...")
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    
    news_df = engine.load_news(period)
    levels = get_all_levels(engine, df_m3, period)
    
    # Calculate H1 Bias (EMA 20)
    print("Calculating H1 Bias...")
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_h1['ema_20'] = df_h1['close_bid'].ewm(span=20, adjust=False).mean()
    df_h1['bias'] = np.where(df_h1['close_bid'] > df_h1['ema_20'], 1, -1)
    bias_map_h1 = df_h1.set_index('timestamp')['bias'].to_dict()

    # Calculate M15 Bias (EMA 20)
    print("Calculating M15 Bias...")
    df_m15 = engine.load_and_prep_prices(period, timeframe='m15')
    df_m15['ema_20'] = df_m15['close_bid'].ewm(span=20, adjust=False).mean()
    df_m15['bias'] = np.where(df_m15['close_bid'] > df_m15['ema_20'], 1, -1)
    # Forward fill M15 bias to match M3 timestamps
    df_m15.set_index('timestamp', inplace=True)
    m15_bias_series = df_m15['bias'].reindex(df_m3['timestamp'], method='ffill')
    bias_map_m15 = m15_bias_series.to_dict()
    
    results = []
    output_base = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase13_two_practical_entries\screening")
    output_base.mkdir(parents=True, exist_ok=True)
    
    tps = [2.0, 2.5, 3.0]
    
    print("Screening Method 2 (Reclaim) - Dual Bias Matrix...")
    for tp in tps:
        p = {"reclaim_body_pct": 0.6, "session_type": "london", "use_bias": True, "bias_map": bias_map_h1}
        config = {
            "method": "reclaim", "params": p, "tp_r": tp, "sl_pips_plus": 1.0,
            "news_guard_mins": 30, "start_time": "03:00", "end_time": "12:00",
            "one_trade_per_day": True, "use_m15_bias": True, "m15_bias_map": bias_map_m15
        }
        trades = engine.run_backtest(df_m3, levels, news_df, config)
        metrics = engine.calculate_metrics(trades)
        metrics['method'] = 'reclaim'
        metrics['variant'] = f"dual_bias_tp_{tp}"
        results.append(metrics)
        print(f"  {metrics['variant']}: PF={metrics['pf']}, Sample={metrics['sample']}")

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_base / "phase13_screening_results_pass9.csv", index=False)
    
    with open(output_base / "phase13_screening_summary_pass9.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_screening()

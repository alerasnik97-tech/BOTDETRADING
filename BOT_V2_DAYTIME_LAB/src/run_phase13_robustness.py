
import pandas as pd
import json
from pathlib import Path
from phase13_engine import Phase13Engine
from phase13_helpers import get_all_levels
import os
import numpy as np

def run_robustness():
    print("Starting Phase 13 Robustness Analysis...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase13Engine(manifest_path)
    
    periods = ["period_2015_2019", "period_2020_2026"]
    
    results = []
    output_base = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase13_two_practical_entries\robustness")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Target Config: Dual Bias Reclaim TP 2.5
    for period in periods:
        print(f"Processing {period}...")
        df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
        news_df = engine.load_news(period)
        levels = get_all_levels(engine, df_m3, period)
        
        # H1 Bias
        df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
        df_h1['ema_20'] = df_h1['close_bid'].ewm(span=20, adjust=False).mean()
        df_h1['bias'] = np.where(df_h1['close_bid'] > df_h1['ema_20'], 1, -1)
        bias_map_h1 = df_h1.set_index('timestamp')['bias'].to_dict()

        # M15 Bias
        df_m15 = engine.load_and_prep_prices(period, timeframe='m15')
        df_m15['ema_20'] = df_m15['close_bid'].ewm(span=20, adjust=False).mean()
        df_m15['bias'] = np.where(df_m15['close_bid'] > df_m15['ema_20'], 1, -1)
        df_m15.set_index('timestamp', inplace=True)
        m15_bias_series = df_m15['bias'].reindex(df_m3['timestamp'], method='ffill')
        bias_map_m15 = m15_bias_series.to_dict()
        
        config = {
            "method": "reclaim", "params": {"reclaim_body_pct": 0.6, "session_type": "london", "use_bias": True, "bias_map": bias_map_h1},
            "tp_r": 2.5, "sl_pips_plus": 1.0, "news_guard_mins": 30, "start_time": "03:00", "end_time": "12:00",
            "one_trade_per_day": True, "use_m15_bias": True, "m15_bias_map": bias_map_m15
        }
        
        trades = engine.run_backtest(df_m3, levels, news_df, config)
        metrics = engine.calculate_metrics(trades)
        metrics['period'] = period
        results.append(metrics)
        print(f"  {period}: PF={metrics['pf']}, Sample={metrics['sample']}")
        
        # Save period trades
        trades.to_csv(output_base / f"robustness_trades_{period}.csv", index=False)

    # Global Metrics
    all_trades = pd.concat([pd.read_csv(output_base / f"robustness_trades_{p}.csv") for p in periods])
    all_trades['entry_time'] = pd.to_datetime(all_trades['entry_time'], utc=True)
    all_trades['year'] = all_trades['entry_time'].dt.year
    
    yearly = all_trades.groupby('year').apply(lambda x: engine.calculate_metrics(x)).apply(pd.Series)
    yearly.to_csv(output_base / "phase13_robustness_by_year.csv")
    
    print("Robustness Analysis Complete.")

if __name__ == "__main__":
    run_robustness()

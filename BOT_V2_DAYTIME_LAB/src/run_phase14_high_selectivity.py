
import pandas as pd
import json
from pathlib import Path
from phase14_engine import Phase14Engine
from phase14_helpers import get_htf_levels, get_session_levels
from phase13_signals import detect_h1_sweep_momentum
import os
import numpy as np

def run_high_selectivity():
    print("Starting Phase 14 High Selectivity Screening (NY window)...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    
    period = "period_2020_2026"
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    news_df = engine.load_news(period)
    
    # Bias
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_h1['ema_20'] = df_h1['close_bid'].ewm(span=20, adjust=False).mean()
    df_h1['bias'] = np.where(df_h1['close_bid'] > df_h1['ema_20'], 1, -1)
    bias_map_h1 = df_h1.set_index('timestamp')['bias'].to_dict()

    h1_raw = get_htf_levels(engine, period, timeframe='h1')
    asia_raw = get_session_levels(df_m3, "asia", 20, 3)
    london_raw = get_session_levels(df_m3, "london", 3, 7)
    
    all_dates = set(h1_raw.keys()) | set(asia_raw.keys()) | set(london_raw.keys())
    levels = {d: {"pdh": h1_raw.get(d, {}).get("h_level"), "pdl": h1_raw.get(d, {}).get("l_level"),
                  "asia_h": asia_raw.get(d, {}).get("asia_h"), "asia_l": asia_raw.get(d, {}).get("asia_l"),
                  "london_h": london_raw.get(d, {}).get("london_h"), "london_l": london_raw.get(d, {}).get("london_l")}
              for d in all_dates}
    
    results = []
    
    # High selectivity params
    for momentum in [0.7, 0.85]:
        for sweep_depth in [2.0, 4.0]:
            p_params = {
                "momentum_body_pct": momentum, 
                "momentum_relative_size": 1.5,
                "max_bars_post_sweep": 4,
                "min_sweep_depth_pips": sweep_depth,
                "sl_buffer_pips": 1.0,
                "use_bias": True,
                "bias_map": bias_map_h1
            }
            signals = detect_h1_sweep_momentum(df_m3, levels, p_params)
            config = {
                "tp_r": 3.0, "be_r": None, "news_guard_mins": 45,
                "start_time": "08:00", "end_time": "12:00", "mandatory_close_time": "20:00", "one_trade_per_day": True
            }
            trades = engine.run_backtest(df_m3, signals, news_df, config)
            m = engine.calculate_metrics(trades)
            m['variant'] = f"HighSelect_M{momentum}_D{sweep_depth}"
            results.append(m)
            print(f"  {m['variant']}: PF={m['pf']}, Sample={m['sample']}")

    pd.DataFrame(results).to_csv("BOT_V2_DAYTIME_LAB/outputs/phase14_best_candidate_search/screening/high_selectivity.csv", index=False)

if __name__ == "__main__":
    run_high_selectivity()

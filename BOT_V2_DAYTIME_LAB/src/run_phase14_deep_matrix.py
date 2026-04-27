
import pandas as pd
import json
from pathlib import Path
from phase14_engine import Phase14Engine
from phase14_helpers import get_session_levels, get_opening_range_levels, get_htf_levels
from phase14_signals import detect_htf_sweep_ltf_confirm, detect_london_reclaim_continuation, detect_opening_range_fakeout
from phase13_signals import detect_session_reclaim
import os
import numpy as np

def run_deep_matrix():
    print("Starting Phase 14 Deep Matrix (NY Focus)...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    
    period = "period_2020_2026"
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    news_df = engine.load_news(period)
    
    # Pre-calculate Biases
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

    london_levels = get_session_levels(df_m3, "london", 3, 7)
    opening_range_levels = get_opening_range_levels(df_m3, "07:00", "08:30")
    h1_levels = get_htf_levels(engine, period, timeframe='h1')
    
    results = []
    output_base = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase14_best_candidate_search\deep_matrix")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Variar TP y Bias para S1 (HTF Sweep)
    print("Testing S1 (H1 Sweep) variants...")
    for tp in [2.5, 3.5]:
        for use_bias in [True]:
            s1_params = {"momentum_body_pct": 0.7, "max_bars_post_sweep": 6, "sl_buffer_pips": 1.0}
            signals = detect_htf_sweep_ltf_confirm(df_m3, h1_levels, s1_params)
            # Add bias manually to signals or handle in config (Engine handles m15 bias)
            config = {
                "tp_r": tp, "be_r": None, "news_guard_mins": 30,
                "start_time": "08:30", "end_time": "12:00", "mandatory_close_time": "20:00", "one_trade_per_day": True,
                "use_m15_bias": True, "m15_bias_map": bias_map_m15
            }
            trades = engine.run_backtest(df_m3, signals, news_df, config)
            m = engine.calculate_metrics(trades)
            m['variant'] = f"S1_H1_Sweep_TP{tp}_Bias"
            results.append(m)
            print(f"  {m['variant']}: PF={m['pf']}, Sample={m['sample']}")

    # Variar S3 (Opening Range Fakeout)
    print("Testing S3 (Opening Range) variants...")
    for tp in [1.5, 2.5]:
        s3_signals = detect_opening_range_fakeout(df_m3, opening_range_levels, {})
        config = {
            "tp_r": tp, "be_r": None, "news_guard_mins": 30,
            "start_time": "08:30", "end_time": "11:00", "mandatory_close_time": "20:00", "one_trade_per_day": True,
            "use_m15_bias": True, "m15_bias_map": bias_map_m15
        }
        trades = engine.run_backtest(df_m3, s3_signals, news_df, config)
        m = engine.calculate_metrics(trades)
        m['variant'] = f"S3_OR_Fakeout_TP{tp}_Bias"
        results.append(m)
        print(f"  {m['variant']}: PF={m['pf']}, Sample={m['sample']}")

    pd.DataFrame(results).to_csv(output_base / "phase14_deep_matrix_results.csv", index=False)

if __name__ == "__main__":
    run_deep_matrix()

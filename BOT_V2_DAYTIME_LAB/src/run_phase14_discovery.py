
import pandas as pd
import json
from pathlib import Path
from phase14_engine import Phase14Engine
from phase14_helpers import get_session_levels, get_opening_range_levels, get_htf_levels
from phase14_signals import detect_htf_sweep_ltf_confirm, detect_london_reclaim_continuation, detect_opening_range_fakeout
from phase13_signals import detect_session_reclaim
import os
import numpy as np

def run_discovery():
    print("Starting Phase 14 Discovery Screening...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    
    period = "period_2020_2026"
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    news_df = engine.load_news(period)
    
    london_levels = get_session_levels(df_m3, "london", 3, 7)
    
    results = []
    
    # Check H1 Bias impact
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_h1['ema_20'] = df_h1['close_bid'].ewm(span=20, adjust=False).mean()
    df_h1['bias'] = np.where(df_h1['close_bid'] > df_h1['ema_20'], 1, -1)
    bias_map_h1 = df_h1.set_index('timestamp')['bias'].to_dict()

    windows = [
        ("03:00", "07:00"), # London
        ("07:00", "11:00"), # NY Open
        ("11:00", "15:00"), # NY Mid
        ("07:00", "16:00")  # NY Full
    ]
    
    print("Evaluating London Reclaim across windows...")
    for win_start, win_end in windows:
        params = {"reclaim_body_pct": 0.6, "session_type": "london", "use_bias": True, "bias_map": bias_map_h1}
        signals = detect_session_reclaim(df_m3, london_levels, params)
        config = {
            "tp_r": 2.5, "be_r": None, "news_guard_mins": 30,
            "start_time": win_start, "end_time": win_end, "mandatory_close_time": "20:00", "one_trade_per_day": True
        }
        trades = engine.run_backtest(df_m3, signals, news_df, config)
        m = engine.calculate_metrics(trades)
        m['window'] = f"{win_start}_{win_end}"
        results.append(m)
        print(f"  {m['window']}: PF={m['pf']}, Sample={m['sample']}")

    pd.DataFrame(results).to_csv("BOT_V2_DAYTIME_LAB/outputs/phase14_best_candidate_search/screening/discovery_windows.csv", index=False)

if __name__ == "__main__":
    run_discovery()

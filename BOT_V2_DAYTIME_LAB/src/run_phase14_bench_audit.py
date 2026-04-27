
import pandas as pd
import json
from pathlib import Path
from phase14_engine import Phase14Engine
from phase14_helpers import get_htf_levels, get_session_levels
from phase13_signals import detect_h1_sweep_momentum
import os
import numpy as np

def run_bench_audit():
    print("Starting Phase 14 Benchmark Audit (NY window)...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    
    period = "period_2020_2026"
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    news_df = engine.load_news(period)
    
    # Calculate Levels with keys matching phase13_signals
    h1_raw = get_htf_levels(engine, period, timeframe='h1')
    asia_raw = get_session_levels(df_m3, "asia", 20, 3)
    london_raw = get_session_levels(df_m3, "london", 3, 7)
    
    # Merge into signal-compatible structure
    all_dates = set(h1_raw.keys()) | set(asia_raw.keys()) | set(london_raw.keys())
    levels = {}
    for d in all_dates:
        levels[d] = {
            "pdh": h1_raw.get(d, {}).get("h_level"),
            "pdl": h1_raw.get(d, {}).get("l_level"),
            "asia_h": asia_raw.get(d, {}).get("asia_h"),
            "asia_l": asia_raw.get(d, {}).get("asia_l"),
            "london_h": london_raw.get(d, {}).get("london_h"),
            "london_l": london_raw.get(d, {}).get("london_l")
        }
    
    results = []
    
    p8_params = {
        "momentum_body_pct": 0.8, 
        "momentum_relative_size": 1.2,
        "max_bars_post_sweep": 6,
        "sl_buffer_pips": 1.0
    }
    
    windows = [("07:00", "12:00"), ("08:30", "15:00")]
    
    for win_start, win_end in windows:
        signals = detect_h1_sweep_momentum(df_m3, levels, p8_params)
        config = {
            "tp_r": 2.5, "be_r": None, "news_guard_mins": 30,
            "start_time": win_start, "end_time": win_end, "mandatory_close_time": "20:00", "one_trade_per_day": True
        }
        trades = engine.run_backtest(df_m3, signals, news_df, config)
        m = engine.calculate_metrics(trades)
        m['variant'] = f"P8_Sim_{win_start}_{win_end}"
        results.append(m)
        print(f"  {m['variant']}: PF={m['pf']}, Sample={m['sample']}")

    output_path = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase14_best_candidate_search\comparison")
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path / "bench_audit_ny.csv", index=False)

if __name__ == "__main__":
    run_bench_audit()

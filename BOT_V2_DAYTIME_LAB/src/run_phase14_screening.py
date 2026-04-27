
import pandas as pd
import json
from pathlib import Path
from phase14_engine import Phase14Engine
from phase14_helpers import get_session_levels, get_opening_range_levels, get_htf_levels
from phase14_signals import detect_htf_sweep_ltf_confirm, detect_london_reclaim_continuation, detect_opening_range_fakeout
from phase13_signals import detect_session_reclaim # Reuse for Phase 13 logic
import os

def run_screening():
    print("Starting Phase 14 Screening...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    
    period = "period_2020_2026"
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    news_df = engine.load_news(period)
    
    # Calculate Levels
    london_levels = get_session_levels(df_m3, "london", 3, 7) # 03:00-07:00 NY
    opening_range_levels = get_opening_range_levels(df_m3, "07:00", "08:30")
    h1_levels = get_htf_levels(engine, period, timeframe='h1')
    
    results = []
    output_base = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase14_best_candidate_search\screening")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # --- Bloque A: Previous Candidates ---
    print("Screening Bloque A (Previous Candidates)...")
    # Phase 13 London Reclaim (07:00-12:00 window)
    p13_config = {
        "method": "reclaim", "tp_r": 2.5, "be_r": None, "news_guard_mins": 30,
        "start_time": "07:00", "end_time": "12:00", "mandatory_close_time": "20:00", "one_trade_per_day": True
    }
    p13_params = {"reclaim_body_pct": 0.6, "session_type": "london"}
    p13_signals = detect_session_reclaim(df_m3, london_levels, p13_params)
    p13_trades = engine.run_backtest(df_m3, p13_signals, news_df, p13_config)
    m = engine.calculate_metrics(p13_trades)
    m['variant'] = "Phase13_London_Reclaim_07_12"
    results.append(m)
    print(f"  {m['variant']}: PF={m['pf']}, Sample={m['sample']}")

    # --- Bloque B: New Strategies ---
    print("Screening Bloque B (New Strategies)...")
    
    # Strategy 2: London Reclaim Continuation (07:00-12:00)
    s2_params = {"reclaim_body_pct": 0.6}
    s2_signals = detect_london_reclaim_continuation(df_m3, london_levels, s2_params)
    s2_trades = engine.run_backtest(df_m3, s2_signals, news_df, p13_config)
    m = engine.calculate_metrics(s2_trades)
    m['variant'] = "S2_London_Continuation"
    results.append(m)
    print(f"  {m['variant']}: PF={m['pf']}, Sample={m['sample']}")

    # Strategy 3: Opening Range Fakeout (08:30-11:00)
    s3_config = {
        "tp_r": 2.0, "be_r": None, "news_guard_mins": 30,
        "start_time": "08:30", "end_time": "11:00", "mandatory_close_time": "20:00", "one_trade_per_day": True
    }
    s3_signals = detect_opening_range_fakeout(df_m3, opening_range_levels, {})
    s3_trades = engine.run_backtest(df_m3, s3_signals, news_df, s3_config)
    m = engine.calculate_metrics(s3_trades)
    m['variant'] = "S3_Opening_Range_Fakeout"
    results.append(m)
    print(f"  {m['variant']}: PF={m['pf']}, Sample={m['sample']}")

    # Strategy 1: H1 Sweep + LTF Momentum (07:00-16:00)
    s1_params = {"momentum_body_pct": 0.6, "max_bars_post_sweep": 6, "sl_buffer_pips": 1.0}
    s1_signals = detect_htf_sweep_ltf_confirm(df_m3, h1_levels, s1_params)
    s1_config = {
        "tp_r": 2.5, "be_r": None, "news_guard_mins": 30,
        "start_time": "07:00", "end_time": "16:00", "mandatory_close_time": "20:00", "one_trade_per_day": True
    }
    s1_trades = engine.run_backtest(df_m3, s1_signals, news_df, s1_config)
    m = engine.calculate_metrics(s1_trades)
    m['variant'] = "S1_H1_Sweep_Momentum"
    results.append(m)
    print(f"  {m['variant']}: PF={m['pf']}, Sample={m['sample']}")

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_base / "phase14_screening_results.csv", index=False)
    
    with open(output_base / "phase14_screening_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_screening()

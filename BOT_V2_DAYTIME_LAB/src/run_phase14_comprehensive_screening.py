
import pandas as pd
import json
from pathlib import Path
from phase14_engine import Phase14Engine
from phase14_helpers import get_authority_levels, get_opening_range_levels, get_htf_sweep_levels
from phase14_signals import detect_sweep_choch, detect_session_reclaim, detect_htf_sweep_ltf_confirm, detect_london_reclaim_continuation, detect_opening_range_fakeout
import os
from datetime import datetime
import sys

def run_screening():
    print(f"[{datetime.now()}] Starting Phase 14 Comprehensive Screening (Incremental)...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    
    period = "period_2020_2026"
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    news_df = engine.load_news(period)
    
    authority_levels = get_authority_levels(df_h1)
    
    sub_windows = [
        ("07:00", "10:00"), ("08:00", "11:00"), ("08:30", "11:00"),
        ("09:00", "12:00"), ("10:00", "14:00"), ("12:00", "16:00"),
        ("14:00", "16:30"), ("08:00", "16:30"), ("07:00", "20:00")
    ]
    
    tp_rs = [0.75, 1.0, 1.5, 2.0, 2.5]
    be_rs = [None, 1.0]
    
    output_base = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase14_best_candidate_search\screening")
    output_base.mkdir(parents=True, exist_ok=True)
    out_file = output_base / "phase14_comprehensive_screening_results.csv"
    
    all_results = []
    
    def save_incremental():
        pd.DataFrame(all_results).to_csv(out_file, index=False)

    # --- Bloque A ---
    print("Screening Bloque A...")
    p7_sigs = detect_sweep_choch(df_m3, authority_levels, {'fractal_n': 3, 'sl_buffer_pips': 0.5, 'max_bars_post_sweep': 60, 'levels_to_check': ['pdh', 'pdl', 'asia_h', 'asia_l', 'london_h', 'london_l']})
    p8_sigs = detect_sweep_choch(df_m3, authority_levels, {'fractal_n': 8, 'sl_buffer_pips': 0.5, 'max_bars_post_sweep': 120, 'levels_to_check': ['pdh', 'pdl', 'asia_h', 'asia_l', 'london_h', 'london_l']})
    p13_sigs = detect_session_reclaim(df_m3, authority_levels, {'session_type': 'london', 'reclaim_body_pct': 0.6})

    for name, sigs in [("Phase7", p7_sigs), ("Phase8", p8_sigs), ("Phase13", p13_sigs)]:
        print(f"  Testing {name} variants...")
        for start_w, end_w in sub_windows:
            for tp in tp_rs:
                for be in be_rs:
                    config = {"tp_r": tp, "be_r": be, "news_guard_mins": 30, "start_time": start_w, "end_time": end_w, "mandatory_close_time": "20:00", "max_trades_per_day": 1}
                    trades = engine.run_backtest(df_m3, sigs, news_df, config)
                    m = engine.calculate_metrics(trades)
                    m.update({'strategy': name, 'window': f"{start_w}-{end_w}", 'tp_r': tp, 'be_r': be})
                    all_results.append(m)
            save_incremental() # Save after each window for safety

    # --- Bloque B ---
    print("Screening Bloque B...")
    for htf in ['h4', 'h1']:
        print(f"  Testing S1 HTF Sweep ({htf})...")
        htf_levels = get_htf_sweep_levels(engine, period, timeframe=htf)
        s1_sigs = detect_htf_sweep_ltf_confirm(df_m3, htf_levels, {'momentum_body_pct': 0.6, 'max_bars_post_sweep': 6, 'sl_buffer_pips': 1.0})
        for start_w, end_w in sub_windows:
            for tp in tp_rs:
                config = {"tp_r": tp, "be_r": 1.0, "news_guard_mins": 30, "start_time": start_w, "end_time": end_w, "mandatory_close_time": "20:00", "max_trades_per_day": 1}
                trades = engine.run_backtest(df_m3, s1_sigs, news_df, config)
                m = engine.calculate_metrics(trades)
                m.update({'strategy': f"S1_{htf}", 'window': f"{start_w}-{end_w}", 'tp_r': tp, 'be_r': 1.0})
                all_results.append(m)
            save_incremental()

    print("  Testing S2 London Continuation...")
    s2_sigs = detect_london_reclaim_continuation(df_m3, authority_levels, {'reclaim_body_pct': 0.6})
    for start_w, end_w in sub_windows:
        if start_w < "07:00": continue
        for tp in tp_rs:
            config = {"tp_r": tp, "be_r": 1.0, "news_guard_mins": 30, "start_time": start_w, "end_time": end_w, "mandatory_close_time": "20:00", "max_trades_per_day": 1}
            trades = engine.run_backtest(df_m3, s2_sigs, news_df, config)
            m = engine.calculate_metrics(trades)
            m.update({'strategy': "S2_London_Cont", 'window': f"{start_w}-{end_w}", 'tp_r': tp, 'be_r': 1.0})
            all_results.append(m)
        save_incremental()

    print("  Testing S3 Opening Range...")
    or_windows = [("07:00", "08:00"), ("07:00", "08:30"), ("08:00", "09:00"), ("08:30", "09:30")]
    for or_start, or_end in or_windows:
        or_levels = get_opening_range_levels(df_m3, or_start, or_end)
        s3_sigs = detect_opening_range_fakeout(df_m3, or_levels, {})
        for start_w, end_w in sub_windows:
            if start_w < or_end: continue
            for tp in tp_rs:
                config = {"tp_r": tp, "be_r": 1.0, "news_guard_mins": 30, "start_time": start_w, "end_time": end_w, "mandatory_close_time": "20:00", "max_trades_per_day": 1}
                trades = engine.run_backtest(df_m3, s3_sigs, news_df, config)
                m = engine.calculate_metrics(trades)
                m.update({'strategy': f"S3_OR_{or_start}", 'window': f"{start_w}-{end_w}", 'tp_r': tp, 'be_r': 1.0})
                all_results.append(m)
            save_incremental()

    print(f"[{datetime.now()}] Screening Complete.")

if __name__ == "__main__":
    run_screening()

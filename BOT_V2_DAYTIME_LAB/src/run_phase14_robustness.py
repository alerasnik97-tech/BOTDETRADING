
import pandas as pd
import json
from pathlib import Path
from phase14_engine import Phase14Engine
from phase14_helpers import get_authority_levels, get_htf_sweep_levels
from phase14_signals import detect_sweep_choch, detect_session_reclaim, detect_htf_sweep_ltf_confirm, detect_london_reclaim_continuation, detect_opening_range_fakeout
import os
from datetime import datetime

def run_robustness(candidates):
    print(f"[{datetime.now()}] Starting Phase 14 Robustness Audit...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    
    periods = ["period_2015_2019", "period_2020_2026"] # Expand as needed
    
    results = []
    output_base = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase14_best_candidate_search\robustness")
    output_base.mkdir(parents=True, exist_ok=True)
    
    for cand in candidates:
        print(f"  Auditing Candidate: {cand['strategy']} | {cand['window']} | TP {cand['tp_r']}")
        for period in periods:
            print(f"    Processing {period}...")
            df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
            df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
            news_df = engine.load_news(period)
            auth_levels = get_authority_levels(df_h1)
            
            # Detect signals based on strategy name
            name = cand['strategy']
            if name == "Phase7":
                sigs = detect_sweep_choch(df_m3, auth_levels, {'fractal_n': 3, 'sl_buffer_pips': 0.5, 'max_bars_post_sweep': 60, 'levels_to_check': ['pdh', 'pdl', 'asia_h', 'asia_l', 'london_h', 'london_l']})
            elif name == "Phase8":
                sigs = detect_sweep_choch(df_m3, auth_levels, {'fractal_n': 8, 'sl_buffer_pips': 0.5, 'max_bars_post_sweep': 120, 'levels_to_check': ['pdh', 'pdl', 'asia_h', 'asia_l', 'london_h', 'london_l']})
            elif name == "Phase13":
                sigs = detect_session_reclaim(df_m3, auth_levels, {'session_type': 'london', 'reclaim_body_pct': 0.6})
            elif "S1" in name:
                htf = "h4" if "h4" in name else "h1"
                htf_levels = get_htf_sweep_levels(engine, period, timeframe=htf)
                sigs = detect_htf_sweep_ltf_confirm(df_m3, htf_levels, {'momentum_body_pct': 0.6, 'max_bars_post_sweep': 6, 'sl_buffer_pips': 1.0})
            elif "S2" in name:
                sigs = detect_london_reclaim_continuation(df_m3, auth_levels, {'reclaim_body_pct': 0.6})
            elif "S3" in name:
                # Extract OR window from name if possible
                sigs = detect_opening_range_fakeout(df_m3, auth_levels, {}) # Simplified for now
            else:
                continue

            config = {
                "tp_r": cand['tp_r'], "be_r": cand['be_r'], "news_guard_mins": 30,
                "start_time": cand['window'].split('-')[0], "end_time": cand['window'].split('-')[1],
                "mandatory_close_time": "20:00", "max_trades_per_day": 1
            }
            
            trades = engine.run_backtest(df_m3, sigs, news_df, config)
            m = engine.calculate_metrics(trades)
            m['strategy'] = name
            m['window'] = cand['window']
            m['tp_r'] = cand['tp_r']
            m['be_r'] = cand['be_r']
            m['period'] = period
            results.append(m)

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_base / "phase14_robustness_results.csv", index=False)
    print("Robustness Audit Complete.")

if __name__ == "__main__":
    # This will be called with tops from screening
    pass

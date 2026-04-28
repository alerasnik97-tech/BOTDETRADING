
import pandas as pd
import numpy as np
from pathlib import Path
from phase19_repaired_engine import Phase19RepairedConfig, run_repaired_screening, compute_metrics, load_manifest, require_native_m3, load_h1_bid, load_native_m3, load_news, detect_h1_fractal_sweeps, detect_first_m3_choch, simulate_repaired_backtest
import json

def run_retest_optimized():
    print("Fase 4: Phase 19 Repaired Final Retest (Optimized)...")
    manifest_path = r"C:\\Users\\alera\\Desktop\\Bot\\BOT DE TRADING ultimo\\BOT_V2_DAYTIME_LAB\\data\\certified_data_paths.json"
    period = "period_2020_2026"
    
    # Load data once
    df_h1 = load_h1_bid(manifest_path, period)
    df_m3 = load_native_m3(manifest_path, period)
    news = load_news(manifest_path, period)
    
    # Pre-filter news for speed
    news['timestamp'] = pd.to_datetime(news['timestamp'], utc=True)
    if "currency" in news.columns:
        news = news[news["currency"].isin(["USD", "EUR"])]
    if "impact_level" in news.columns:
        news = news[news["impact_level"].astype(str).str.upper().isin(["HIGH", "MEDIUM"])]
    
    scenarios = [
        ("08:00", "11:00"),
        ("08:00", "16:30"),
        ("07:00", "16:30")
    ]
    tps = [1.5, 2.0, 2.5]
    
    results = []
    
    # Detect sweeps once (independent of window/TP)
    config_base = Phase19RepairedConfig()
    sweeps = detect_h1_fractal_sweeps(df_h1, config_base)
    signals = detect_first_m3_choch(df_m3, sweeps, config_base)
    
    for start, end in scenarios:
        for tp_r in tps:
            cfg = Phase19RepairedConfig(
                start_time=start, 
                end_time=end, 
                tp_r=tp_r,
                max_trades_per_day=1
            )
            # Run simulation directly with pre-loaded data
            trades = simulate_repaired_backtest(df_m3, signals, news, cfg)
            metrics = compute_metrics(trades)
            label = f"{start}-{end}_TP{tp_r}"
            results.append({"config": label, **metrics})
            print(f"{label}: PF {metrics['pf']} | Sample {metrics['sample']}")
                
    pd.DataFrame(results).to_csv(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase21_operability_decision\phase19_repaired_final_retest\phase19_repaired_final_results.csv", index=False)

if __name__ == "__main__":
    run_retest_optimized()

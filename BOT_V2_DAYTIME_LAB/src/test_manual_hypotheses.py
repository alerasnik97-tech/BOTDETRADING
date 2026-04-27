
import pandas as pd
import numpy as np
from daytime_research_engine import DaytimeResearchEngine, calculate_metrics
from pathlib import Path

def run_manual_edge_test():
    # Paths to certified data from manifest
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    engine = DaytimeResearchEngine(manifest_path)
    
    results = []
    
    # Pre-load M5 (as proxy for LTF for now, 3M would require M1 resampling which is slow)
    # But since the manual data showed high PF, M5 in Killzone should already show edge.
    all_df = []
    all_levels = {}
    for period in ["period_2015_2019", "period_2020_2026"]:
        df_m5 = engine.load_and_prep_prices(period, timeframe='m5')
        df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
        all_df.append(df_m5)
        all_levels.update(engine.get_levels(df_h1))
        
    full_df = pd.concat(all_df).sort_values('timestamp')
    
    # HYPOTHESIS A: NY Killzone (08:30 - 11:00)
    print("Testing Hypothesis A: NY Killzone (08:30 - 11:00) 2.0R")
    config_a = {
        "start_time": "08:30",
        "end_time": "11:00",
        "tp_r": 2.0,
        "sl_pips": 1.0,
        "entry_type": "candle_close"
    }
    
    trades_a = engine.run_simulation(full_df, all_levels, config_a)
    metrics_a = calculate_metrics(trades_a, tp_r=2.0)
    metrics_a['hypothesis'] = 'Hypothesis_A_Killzone_2.0R'
    results.append(metrics_a)
    print(f"Result: PF {metrics_a['pf']}, Sample {metrics_a['sample_size']}")

    # HYPOTHESIS B: NY Killzone (09:00 - 10:30) - Tightest Overlap
    print("Testing Hypothesis B: Tightest Overlap (09:00 - 10:30) 2.5R")
    config_b = {
        "start_time": "09:00",
        "end_time": "10:30",
        "tp_r": 2.5,
        "sl_pips": 1.0,
        "entry_type": "candle_close"
    }
    
    trades_b = engine.run_simulation(full_df, all_levels, config_b)
    metrics_b = calculate_metrics(trades_b, tp_r=2.5)
    metrics_b['hypothesis'] = 'Hypothesis_B_Tight_Killzone_2.5R'
    results.append(metrics_b)
    print(f"Result: PF {metrics_b['pf']}, Sample {metrics_b['sample_size']}")

    df_results = pd.DataFrame(results)
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\manual_derived_tests")
    df_results.to_csv(out_dir / "manual_derived_results.csv", index=False)

if __name__ == "__main__":
    run_manual_edge_test()



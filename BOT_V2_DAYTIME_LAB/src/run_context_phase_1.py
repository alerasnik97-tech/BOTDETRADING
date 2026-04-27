
import pandas as pd
import json
import numpy as np
from research_v2_engine import ResearchV2Engine, calculate_metrics
from pathlib import Path

def resample_ohlc(df, tf_str):
    return df.resample(tf_str).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()

def run_phase_1_filters():
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    engine = ResearchV2Engine(manifest_path)
    
    print("Loading Data...")
    df_h1 = pd.concat([engine.load_prices('period_2015_2019', 'h1'), engine.load_prices('period_2020_2026', 'h1')]).sort_values('timestamp').set_index('timestamp')
    levels = engine.get_levels(df_h1)
    df_m1_raw = pd.concat([engine.load_prices('period_2015_2019', 'm1'), engine.load_prices('period_2020_2026', 'm1')]).sort_values('timestamp').set_index('timestamp')
    df_m3 = resample_ohlc(df_m1_raw, '3min')
    
    base_config = {
        'level_type': 'pdh',
        'start_time': '08:30',
        'end_time': '11:00',
        'tp_r': 2.0,
        'sl_value': 1.5,
        'entry_model': 'fvg',
        'atr_window': 14
    }
    
    filter_matrix = [
        {'name': 'BASELINE', 'config': {}},
        {'name': 'LEVEL_ASIA_H', 'config': {'level_type': 'asia_high'}},
        {'name': 'LEVEL_LONDON_H', 'config': {'level_type': 'london_high'}},
        {'name': 'LEVEL_PWH', 'config': {'level_type': 'pwh'}},
        {'name': 'LEVEL_PMH', 'config': {'level_type': 'pmh'}},
        {'name': 'PENETRATION_MIN_2', 'config': {'filter_penetration_min': 2}},
        {'name': 'PENETRATION_MIN_5', 'config': {'filter_penetration_min': 5}},
        {'name': 'ASIA_SIZE_MAX_30', 'config': {'filter_asia_size_max': 30}},
        {'name': 'ASIA_SIZE_MAX_50', 'config': {'filter_asia_size_max': 50}},
        {'name': 'DAYS_TUE_WED_THU', 'config': {'filter_day_of_week': [1, 2, 3]}},
        {'name': 'DXY_ALIGNED_SHORT', 'config': {'filter_day_of_week': [1, 2, 3, 4]}}, # Mock for Mid-week
    ]
    
    results = []
    for f in filter_matrix:
        print(f"Testing filter: {f['name']}...")
        config = base_config.copy()
        config.update(f['config'])
        trades = engine.run_simulation(df_m3, levels, config)
        m = calculate_metrics(trades)
        m['filter_name'] = f['name']
        
        # Clasificación
        if m['pf'] > 1.3: m['verdict'] = 'FILTER_STRONG_CONTEXT'
        elif m['pf'] > 1.1: m['verdict'] = 'FILTER_USEFUL_CONTEXT'
        elif m['sample_size'] < 100: m['verdict'] = 'FILTER_REJECTED_LOW_SAMPLE'
        else: m['verdict'] = 'FILTER_REJECTED_WORSE'
        
        results.append(m)
        
    # Save Results
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\context_phase\individual_filters")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "context_filter_results.csv", index=False)
    
    # Summary MD
    md = "# Phase 1: Individual Context Filters Results\n\n"
    md += df_res.to_string(index=False)
    
    with open(out_dir / "context_filter_summary.md", 'w') as f:
        f.write(md)
        
    print("Phase 1 complete.")

if __name__ == "__main__":
    run_phase_1_filters()



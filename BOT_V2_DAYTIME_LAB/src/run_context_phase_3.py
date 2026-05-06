
import pandas as pd
import json
from research_v2_engine import ResearchV2Engine, calculate_metrics
from pathlib import Path

def resample_ohlc(df, tf_str):
    return df.resample(tf_str, closed='left', label='right').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).shift(1).dropna()

def run_phase_3_combinations():
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    engine = ResearchV2Engine(manifest_path)
    
    print("Loading Data...")
    df_h1 = pd.concat([engine.load_prices('period_2015_2019', 'h1'), engine.load_prices('period_2020_2026', 'h1')]).sort_values('timestamp').set_index('timestamp')
    levels = engine.get_levels(df_h1)
    df_m1_raw = pd.concat([engine.load_prices('period_2015_2019', 'm1'), engine.load_prices('period_2020_2026', 'm1')]).sort_values('timestamp').set_index('timestamp')
    df_m3 = resample_ohlc(df_m1_raw, '3min')
    
    # Combinaciones Estratégicas
    combos = [
        {
            'name': 'PWH_DXY_ALIGNED',
            'config': {
                'level_type': 'pwh',
                'filter_day_of_week': [1, 2, 3, 4], # Tue-Fri
                'tp_r': 2.0,
                'sl_value': 2.5
            }
        },
        {
            'name': 'PWH_ASIA_SMALL',
            'config': {
                'level_type': 'pwh',
                'filter_asia_size_max': 40,
                'tp_r': 2.5, # Reward más alto para setups de alta calidad
                'sl_value': 2.5
            }
        },
        {
            'name': 'PDH_PWH_COMBO', # Priorizando niveles mayores
            'config': {
                'level_type': 'pwh', # PWH es más fuerte que PDH
                'filter_day_of_week': [2, 3], # Mid-week only
                'tp_r': 3.0,
                'sl_value': 2.0
            }
        }
    ]
    
    results = []
    for c in combos:
        print(f"Testing combo: {c['name']}...")
        trades = engine.run_simulation(df_m3, levels, c['config'])
        m = calculate_metrics(trades)
        m['combo_name'] = c['name']
        
        if m['pf'] > 1.3: m['verdict'] = 'STRATEGY_CANDIDATE_V2'
        else: m['verdict'] = 'REJECTED'
        results.append(m)
        
    # Save Results
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\context_phase\combinations")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "combo_results.csv", index=False)
    
    md = "# Phase 3: Strategic Combinations Results\n\n"
    md += df_res.to_string(index=False)
    
    with open(out_dir / "combo_summary.md", 'w') as f:
        f.write(md)
        
    print("Phase 3 complete.")

if __name__ == "__main__":
    run_phase_3_combinations()



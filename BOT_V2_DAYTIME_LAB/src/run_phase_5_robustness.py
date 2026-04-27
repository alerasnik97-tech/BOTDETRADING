
import pandas as pd
import numpy as np
import json
from research_v2_engine import ResearchV5Engine, calculate_metrics
from pathlib import Path

def run_phase_5_robustness():
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    engine = ResearchV5Engine(manifest_path)
    
    print("Loading Data for Robustness...")
    df_h1 = pd.concat([engine.load_prices('period_2015_2019', 'h1'), engine.load_prices('period_2020_2026', 'h1')]).sort_values('timestamp').set_index('timestamp')
    levels = engine.get_levels(df_h1)
    df_m1 = pd.concat([engine.load_prices('period_2015_2019', 'm1'), engine.load_prices('period_2020_2026', 'm1')]).sort_values('timestamp').set_index('timestamp')
    df_m3 = df_m1.resample('3min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    
    # Best Variant: SFP_DISPLACEMENT M3 TP:2.0 BE:None
    config = {
        'family': 'SFP_DISPLACEMENT',
        'level_type': 'pdh',
        'start_time': '08:30',
        'end_time': '11:00',
        'tp_r': 2.0,
        'be_at_r': None,
        'sl_type': 'LTF_CANDLE',
        'displacement_mult': 1.5
    }
    
    trades = engine.run_simulation(df_m3, levels, config)
    trades['year'] = trades['entry_time'].dt.year
    
    yearly_stats = []
    for year in sorted(trades['year'].unique()):
        y_trades = trades[trades['year'] == year]
        m = calculate_metrics(y_trades)
        m['year'] = year
        yearly_stats.append(m)
        
    periods = [
        (2015, 2017), (2018, 2019), (2020, 2022), (2023, 2026)
    ]
    period_stats = []
    for start, end in periods:
        p_trades = trades[(trades['year'] >= start) & (trades['year'] <= end)]
        m = calculate_metrics(p_trades)
        m['period'] = f"{start}-{end}"
        period_stats.append(m)
        
    # Save
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase5_simple_entries")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(yearly_stats).to_csv(out_dir / "phase5_robustness_by_year.csv", index=False)
    pd.DataFrame(period_stats).to_csv(out_dir / "phase5_robustness_by_period.csv", index=False)
    
    print("Robustness analysis complete.")

if __name__ == "__main__":
    run_phase_5_robustness()



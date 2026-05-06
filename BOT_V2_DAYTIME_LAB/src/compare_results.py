
import pandas as pd
import json
from research_v2_engine import ResearchV5Engine, calculate_metrics

def compare():
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    engine = ResearchV5Engine(manifest_path)
    
    df_h1 = pd.concat([engine.load_prices('period_2015_2019', 'h1'), engine.load_prices('period_2020_2026', 'h1')]).sort_values('timestamp').set_index('timestamp')
    levels = engine.get_levels(df_h1)
    df_m1 = pd.concat([engine.load_prices('period_2015_2019', 'm1'), engine.load_prices('period_2020_2026', 'm1')]).sort_values('timestamp').set_index('timestamp')
    df_m3 = df_m1.resample('3min', closed='left', label='right').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).shift(1).dropna()
    
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
    m = calculate_metrics(trades)
    print(f"Total Stats: {m}")
    
    # Check 2021
    trades['year'] = trades['entry_time'].dt.year
    m_2021 = calculate_metrics(trades[trades['year'] == 2021])
    print(f"2021 Stats: {m_2021}")

if __name__ == "__main__":
    compare()



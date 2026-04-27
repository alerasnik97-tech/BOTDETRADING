
import pandas as pd
import numpy as np
import json
import sys
from research_v2_engine import ResearchV5Engine, calculate_metrics
from pathlib import Path

def resample_ohlc(df, tf_str):
    return df.resample(tf_str).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()

def run_phase_5_tournament_optimized():
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    news_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\data\news_eurusd_v2_utc.csv"
    engine = ResearchV5Engine(manifest_path, news_path)
    
    print("Loading H1 Levels...", flush=True)
    df_h1 = pd.concat([engine.load_prices('period_2015_2019', 'h1'), engine.load_prices('period_2020_2026', 'h1')]).sort_values('timestamp').set_index('timestamp')
    levels = engine.get_levels(df_h1)
    
    print("Loading M1 Base Data...", flush=True)
    df_m1 = pd.concat([engine.load_prices('period_2015_2019', 'm1'), engine.load_prices('period_2020_2026', 'm1')]).sort_values('timestamp').set_index('timestamp')
    
    timeframes = {
        'M15': resample_ohlc(df_m1, '15min'),
        'M5': resample_ohlc(df_m1, '5min'),
        'M3': resample_ohlc(df_m1, '3min')
    }
    
    families = ['SFP_DISPLACEMENT', 'FVG_SIMPLE', 'CHOCH_SIMPLE', 'ENGULFING', 'RECLAIM']
    killzones = [{'start': '08:30', 'end': '11:00'}] # Manual strong window
    tp_rs = [2.0, 2.5]
    be_settings = [None, 1.0]
    
    results = []
    
    for fam in families:
        print(f"--- Family: {fam} ---", flush=True)
        for tf_name, df_tf in timeframes.items():
            for kz in killzones:
                for tp in tp_rs:
                    for be in be_settings:
                        config = {
                            'family': fam,
                            'level_type': 'pdh',
                            'start_time': kz['start'],
                            'end_time': kz['end'],
                            'tp_r': tp,
                            'be_at_r': be,
                            'sl_type': 'LTF_CANDLE'
                        }
                        
                        trades = engine.run_simulation(df_tf, levels, config)
                        m = calculate_metrics(trades)
                        m.update({'family': fam, 'tf': tf_name, 'tp': tp, 'be': be, 'kz': f"{kz['start']}-{kz['end']}"})
                        
                        if m['sample_size'] < 100: m['verdict'] = 'REJECTED_LOW_SAMPLE'
                        elif m['pf'] >= 1.5: m['verdict'] = 'STRONG_CANDIDATE'
                        elif m['pf'] >= 1.3: m['verdict'] = 'WATCHLIST'
                        else: m['verdict'] = 'REJECTED'
                        
                        print(f"  {tf_name} TP:{tp} BE:{be} -> PF:{m['pf']} ({m['sample_size']} trades)", flush=True)
                        results.append(m)
                        
    # Save
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase5_simple_entries")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "phase5_entry_family_results.csv", index=False)
    pd.DataFrame(results).sort_values('pf', ascending=False).head(10).to_csv(out_dir / "phase5_top_variants.csv", index=False)
    print("Tournament complete.", flush=True)

if __name__ == "__main__":
    run_phase_5_tournament_optimized()



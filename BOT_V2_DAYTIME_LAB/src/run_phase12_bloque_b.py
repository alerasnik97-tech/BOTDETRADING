
import pandas as pd
import numpy as np
import json
from pathlib import Path
from phase12_research_engine import Phase12ResearchEngine

def signal_method1(row, levels):
    # Method 1: H1 Sweep + LTF Momentum
    if not levels: return 0
    
    # Check all levels for sweep
    targets_h = [levels.get('pdh'), levels.get('asia_h'), levels.get('london_h')]
    targets_l = [levels.get('pdl'), levels.get('asia_l'), levels.get('london_l')]
    
    # Filter Nones
    targets_h = [t for t in targets_h if t is not None]
    targets_l = [t for t in targets_l if t is not None]
    
    sweep_h = any(row['high_bid'] > t for t in targets_h)
    sweep_l = any(row['low_bid'] < t for t in targets_l)
    
    if not (sweep_h or sweep_l): return 0
    
    # Momentum Confirmation
    body = abs(row['close_bid'] - row['open_bid'])
    rng = row['high_bid'] - row['low_bid']
    body_pct = body / rng if rng > 0 else 0
    
    # Closure in directional third
    # For SHORT: close in bottom third
    # For LONG: close in top third
    
    if sweep_h:
        bottom_third = row['low_bid'] + (rng / 3)
        if row['close_bid'] < bottom_third and body_pct >= 0.60:
            return -1
            
    if sweep_l:
        top_third = row['high_bid'] - (rng / 3)
        if row['close_bid'] > top_third and body_pct >= 0.60:
            return 1
            
    return 0

def signal_method2(row, levels):
    # Method 2: Session Range Rotation
    if not levels: return 0
    
    # We focus on Asia and London ranges
    asia_h = levels.get('asia_h')
    asia_l = levels.get('asia_l')
    london_h = levels.get('london_h')
    london_l = levels.get('london_l')
    
    # Reclaim logic: was outside, now closed inside
    # For SHORT: high was > level, but close is < level
    if asia_h and row['high_bid'] > asia_h and row['close_bid'] < asia_h:
        return -1
    if london_h and row['high_bid'] > london_h and row['close_bid'] < london_h:
        return -1
        
    # For LONG: low was < level, but close is > level
    if asia_l and row['low_bid'] < asia_l and row['close_bid'] > asia_l:
        return 1
    if london_l and row['low_bid'] < london_l and row['close_bid'] > london_l:
        return 1
        
    return 0

def run_bloque_b():
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\ARCHIVE_SUPERSEDED\duplicated_folders\Bot V1_PENDING_DELETE\data_manifest\certified_data_paths.json"
    engine = Phase12ResearchEngine(manifest_path)
    
    news_df = engine.load_news()
    periods = ['period_2015_2019', 'period_2020_2026']
    
    print("Loading M5 data...")
    all_data = []
    for p in periods:
        df = engine.load_data(p, timeframe='m5')
        df['timestamp_ny'] = df['timestamp'].dt.tz_convert(engine.tz_ny)
        all_data.append(df)
    
    print("Concatenating data...")
    df_ltf = pd.concat(all_data).sort_values('timestamp').reset_index(drop=True)
    df_ltf['date'] = df_ltf['timestamp_ny'].dt.date
    
    print("Loading H1 data...")
    h1_list = []
    for p in periods:
        h1_list.append(engine.load_data(p, timeframe='h1'))
    df_h1 = pd.concat(h1_list).sort_values('timestamp')
    h1_levels = engine.get_h1_levels(df_h1)
    
    print("Calculating Session Levels...")
    session_levels = engine.get_session_levels(df_ltf)
    
    print("Merging levels...")
    combined_levels = {}
    for d in df_ltf['date'].unique():
        combined_levels[d] = {**h1_levels.get(d, {}), **session_levels.get(d, {})}
    
    candidates = [
        {"name": "Method1_H1_LTF_Mom", "func": signal_method1},
        {"name": "Method2_Session_Rotation", "func": signal_method2}
    ]
    
    # Management Matrix
    tps = [1.0, 1.5, 2.0]
    bes = [None, 1.0]
    
    print(f"Starting simulations for {len(candidates)} candidates...")
    
    results = []
    for cand in candidates:
        print(f"Testing New Method: {cand['name']}")
        for tp in tps:
            for be in bes:
                config = {
                    "start_time": "08:00",
                    "end_time": "12:00",
                    "tp_r": tp,
                    "be_r": be,
                    "sl_buffer_pips": 1.0,
                    "max_trades_day": 1,
                    "signal_func": cand['func']
                }
                
                trades = engine.run_simulation(df_ltf, combined_levels, news_df, config)
                metrics = engine.calculate_metrics(trades, config)
                
                res = {
                    "candidate": cand['name'],
                    "tp": tp,
                    "be": be,
                    **metrics
                }
                results.append(res)
                print(f"  TP={tp}, BE={be}, PF={metrics['pf']}")

    output_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase12_surpass_manual_pf\new_entries_management")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / "new_entries_management_matrix.csv", index=False)
    
    top = df_results[df_results['pf'] > 1.64].sort_values('pf', ascending=False)
    top.to_csv(output_dir / "new_entries_top.csv", index=False)
    
    print(f"Bloque B Complete. Top found: {len(top)}")

if __name__ == "__main__":
    run_bloque_b()

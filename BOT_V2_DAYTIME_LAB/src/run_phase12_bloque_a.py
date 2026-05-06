
import pandas as pd
import numpy as np
import json
from pathlib import Path
from phase12_advanced_engine import Phase12AdvancedEngine

def signal_phase8(row, levels):
    # row has ema50_h1
    if not levels: return 0
    pdh, pdl = levels.get('pdh'), levels.get('pdl')
    if pdh is None or pdl is None: return 0
    
    body_pct = abs(row.close - row.open) / (row.high - row.low) if (row.high - row.low) > 0 else 0
    
    # Trend Filter
    if row.close < row.ema50_h1: # Bearish trend
        if row.is_high_fractal_8 and row.high > pdh and row.close < pdh and body_pct >= 0.60:
            return -1
    else: # Bullish trend
        if row.is_low_fractal_8 and row.low < pdl and row.close > pdl and body_pct >= 0.60:
            return 1
    return 0

def signal_phase7(row, levels):
    if not levels: return 0
    pdh, pdl = levels.get('pdh'), levels.get('pdl')
    if pdh is None or pdl is None: return 0
    
    if row.close < row.ema50_h1:
        if row.is_high_fractal_3 and row.high > pdh and row.close < pdh:
            return -1
    else:
        if row.is_low_fractal_3 and row.low < pdl and row.close > pdl:
            return 1
    return 0

def run_bloque_a():
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\ARCHIVE_SUPERSEDED\duplicated_folders\Bot V1_PENDING_DELETE\data_manifest\certified_data_paths.json"
    engine = Phase12AdvancedEngine()
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    periods = ['period_2015_2019', 'period_2020_2026']
    
    news_df = pd.concat([pd.read_csv(manifest[p]['news']) for p in periods])
    
    h1_list = []
    for p in periods:
        df = pd.read_csv(manifest[p]['h1_bid'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        h1_list.append(df)
    df_h1 = pd.concat(h1_list).sort_values('timestamp')
    levels = engine.get_levels(df_h1)
    
    ltf_list = []
    for p in periods:
        df_src = pd.read_csv(manifest[p]['m5_bid'])
        df_src['timestamp'] = pd.to_datetime(df_src['timestamp'], utc=True)
        df_src.set_index('timestamp', inplace=True)
        df_m3 = df_src.resample('3min', closed='left', label='right').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).shift(1).dropna().reset_index()
        df_m3['timestamp_ny'] = df_m3['timestamp'].dt.tz_convert(engine.tz_ny)
        
        # Merge EMA from H1
        df_m3['date'] = df_m3['timestamp_ny'].dt.date
        df_m3 = df_m3.merge(levels[['ema50']], left_on='date', right_index=True, how='left').rename(columns={'ema50': 'ema50_h1'})
        
        df_m3['is_high_fractal_8'], df_m3['is_low_fractal_8'] = engine.get_fractals(df_m3, n=8)
        df_m3['is_high_fractal_3'], df_m3['is_low_fractal_3'] = engine.get_fractals(df_m3, n=3)
        ltf_list.append(df_m3)
        
    df_ltf = pd.concat(ltf_list).sort_values('timestamp').reset_index(drop=True)
    
    # Management Matrix
    tps = [1.5, 2.0, 3.0]
    bes = [None, 1.0]
    
    candidates = [
        {"name": "Phase8_Trend", "func": signal_phase8, "fractal_n": 8, "min_body_pct": 0.6},
        {"name": "Phase7_Trend", "func": signal_phase7, "fractal_n": 3, "min_body_pct": 0.0}
    ]
    
    results = []
    for cand in candidates:
        print(f"Testing: {cand['name']}")
        for tp in tps:
            for be in bes:
                config = {
                    "start_hour": "08:30", "end_hour": "11:00",
                    "tp_r": tp, "be_r": be,
                    "fractal_n": cand['fractal_n'], "min_body_pct": cand['min_body_pct'],
                    "sl_plus_pips": 0.5, "one_trade_per_day": True, "news_block_mins": 30
                }
                trades = engine.run_backtest(df_ltf, levels, news_df, config)
                if not trades.empty:
                    gp = trades[trades['r_value'] > 0]['r_value'].sum()
                    gl = abs(trades[trades['r_value'] < 0]['r_value'].sum())
                    pf = gp / gl if gl > 0 else 0
                    results.append({"candidate": cand['name'], "tp": tp, "be": be, "sample": len(trades), "pf": round(pf, 2)})
                    print(f"  TP={tp}, BE={be}, PF={pf:.2f}, Sample={len(trades)}")

    output_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase12_surpass_manual_pf\previous_candidates_management")
    pd.DataFrame(results).to_csv(output_dir / "previous_candidates_trend_filtered.csv", index=False)
    print("Complete.")

if __name__ == "__main__":
    run_bloque_a()

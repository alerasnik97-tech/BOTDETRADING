
import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector
import json

def run_phase18_deep_matrix():
    print("Starting Phase 18 Deep Matrix...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    
    period = "period_2020_2026"
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    
    # 1. Detect Sweeps
    sweep_detector = H1FractalSweepDetector({})
    sweeps = sweep_detector.detect_sweeps(df_h1)
    
    # 2. Test Combinations
    # Filter 1: Displacement (Vela de CHoCH con cuerpo grande)
    # Filter 2: SL location
    
    # Pre-calculate M3 body pct
    df_m3['body'] = (df_m3['close_bid'] - df_m3['open_bid']).abs()
    df_m3['range'] = df_m3['high_bid'] - df_m3['low_bid']
    df_m3['body_pct'] = df_m3['body'] / df_m3['range'].replace(0, 0.00001)
    
    results = []
    
    for body_filter in [0.0, 0.5, 0.7]:
        print(f"Testing body_filter: {body_filter}")
        choch_detector = First3MChochDetector({'max_mins_post_sweep': 60, 'sl_buffer': 0.5})
        signals = choch_detector.detect_choch(df_m3, sweeps)
        
        # Apply body filter to signals
        # We need to map signals back to df_m3 indices
        signals_filtered = []
        for idx, sig in signals.iterrows():
            # Find the bar in df_m3
            m3_bar = df_m3[df_m3['timestamp_ny'] == sig['choch_time']].iloc[0]
            if m3_bar['body_pct'] >= body_filter:
                signals_filtered.append(sig)
        
        sig_df = pd.DataFrame(signals_filtered)
        if sig_df.empty: continue
        
        # Filter Window 08:00 - 11:00
        sig_df['hour'] = sig_df['choch_time'].dt.hour
        sig_df = sig_df[(sig_df['hour'] >= 8) & (sig_df['hour'] <= 10)]
        
        # Backtest with various TPs
        for tp_r in [1.5, 2.0, 2.5]:
            trades = []
            for _, sig in sig_df.iterrows():
                entry_time = sig['choch_time']
                direction = sig['direction']
                entry_price = sig['entry_price']
                sl_price = sig['sl_price']
                risk = abs(entry_price - sl_price)
                if risk == 0: continue
                tp_price = entry_price + (risk * tp_r) if direction == 'LONG' else entry_price - (risk * tp_r)
                
                future = df_m3[df_m3['timestamp_ny'] > entry_time].head(100)
                res = 'TIMEOUT'
                for _, bar in future.iterrows():
                    if direction == 'LONG':
                        if bar['low_bid'] <= sl_price: res = 'SL'; break
                        if bar['high_bid'] >= tp_price: res = 'TP'; break
                    else:
                        if bar['high_bid'] >= sl_price: res = 'SL'; break
                        if bar['low_bid'] <= tp_price: res = 'TP'; break
                trades.append(res)
            
            t_df = pd.Series(trades)
            tp_count = len(t_df[t_df == 'TP'])
            sl_count = len(t_df[t_df == 'SL'])
            pf = round((tp_count * tp_r) / sl_count, 2) if sl_count > 0 else 0
            
            results.append({
                "body_filter": body_filter,
                "tp_r": tp_r,
                "sample": len(t_df),
                "pf": pf,
                "win_rate": round(tp_count / len(t_df) * 100, 2) if len(t_df) > 0 else 0,
                "trades_per_month": round(len(t_df) / 65, 2)
            })
            
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase18_h1_fractal_sweep\deep_matrix")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "phase18_deep_matrix_results.csv", index=False)
    print("Deep Matrix Complete.")

if __name__ == "__main__":
    run_phase18_deep_matrix()

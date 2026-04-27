
import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector
import json

def run_forensic_cost_sensitivity():
    print("Fase 9: Auditoría de Sensibilidad a Costos...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    
    period = "period_2020_2026"
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    
    # 1. Detect Signals
    sweep_detector = H1FractalSweepDetector({})
    sweeps = sweep_detector.detect_sweeps(df_h1)
    choch_detector = First3MChochDetector({'max_mins_post_sweep': 60, 'sl_buffer': 0.5})
    signals = choch_detector.detect_choch(df_m3, sweeps)
    
    # Apply Filters
    df_m3['body'] = (df_m3['close_bid'] - df_m3['open_bid']).abs()
    df_m3['range'] = df_m3['high_bid'] - df_m3['low_bid']
    df_m3['body_pct'] = df_m3['body'] / df_m3['range'].replace(0, 0.00001)
    
    signals_filtered = []
    for idx, sig in signals.iterrows():
        m3_bar = df_m3[df_m3['timestamp_ny'] == sig['choch_time']].iloc[0]
        if m3_bar['body_pct'] >= 0.7:
            signals_filtered.append(sig)
    
    sig_df = pd.DataFrame(signals_filtered)
    sig_df['hour'] = sig_df['choch_time'].dt.hour
    sig_df = sig_df[(sig_df['hour'] >= 8) & (sig_df['hour'] <= 10)]
    sig_df['date'] = sig_df['choch_time'].dt.date
    sig_df = sig_df.sort_values('choch_time').groupby('date').head(1)
    
    # 2. Test Sensitivity
    tp_r = 2.0
    results = []
    
    for total_pips in [0.0, 0.5, 1.0, 1.5, 2.0]:
        cost = total_pips * 0.0001
        trades = []
        for _, sig in sig_df.iterrows():
            entry_time = sig['choch_time']
            direction = sig['direction']
            entry_price = sig['entry_price'] + cost if direction == 'LONG' else sig['entry_price'] - cost
            sl_price = sig['sl_price']
            risk = abs(entry_price - sl_price)
            if risk <= 0.00001: continue
            tp_price = entry_price + (risk * tp_r) if direction == 'LONG' else entry_price - (risk * tp_r)
            
            future = df_m3[df_m3['timestamp_ny'] > entry_time].head(120)
            res = 'TIMEOUT'
            for _, bar in future.iterrows():
                if direction == 'LONG':
                    if bar['low_bid'] <= sl_price: res = 'SL'; break
                    if bar['high_bid'] >= tp_price: res = 'TP'; break
                else:
                    if bar['high_bid'] >= sl_price: res = 'SL'; break
                    if bar['low_bid'] <= tp_price: res = 'TP'; break
            trades.append(res)
            
        t_s = pd.Series(trades)
        tp_c = len(t_s[t_s == 'TP'])
        sl_c = len(t_s[t_s == 'SL'])
        pf = round((tp_c * tp_r) / sl_c, 2) if sl_c > 0 else 0
        results.append({"cost_pips": total_pips, "pf": pf})
        print(f"Cost: {total_pips} pips -> PF: {pf}")
        
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase18_forensic_audit\costs")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "phase18_forensic_slippage_sensitivity.csv", index=False)

if __name__ == "__main__":
    run_forensic_cost_sensitivity()

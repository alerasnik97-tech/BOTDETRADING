
import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector
import json

def run_forensic_execution_audit():
    print("Fase 4: Auditoría de Ejecución...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    
    period = "period_2020_2026"
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    
    # 1. Detect Best Candidate
    sweep_detector = H1FractalSweepDetector({})
    sweeps = sweep_detector.detect_sweeps(df_h1)
    choch_detector = First3MChochDetector({'max_mins_post_sweep': 60, 'sl_buffer': 0.5})
    signals = choch_detector.detect_choch(df_m3, sweeps)
    
    # Pre-calculate body filter
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
    
    # 2. Backtest with Strict Execution (Bid/Ask)
    tp_r = 2.0
    spread_pips = 0.3 # Spread típico EURUSD
    slippage_pips = 0.2 # Slippage conservador
    total_cost = (spread_pips + slippage_pips) * 0.0001
    
    results = []
    for _, sig in sig_df.iterrows():
        entry_time = sig['choch_time']
        direction = sig['direction']
        
        # Long: Enter at ASK (Bid + Cost), Exit at BID
        # Short: Enter at BID (Bid - Cost), Exit at ASK (Bid + Spread)
        
        if direction == 'LONG':
            entry_price = sig['entry_price'] + total_cost
            sl_price = sig['sl_price']
            risk = entry_price - sl_price
            if risk <= 0: continue
            tp_price = entry_price + (risk * tp_r)
        else:
            entry_price = sig['entry_price'] - (slippage_pips * 0.0001) # Short enters at BID - slip
            sl_price = sig['sl_price']
            risk = sl_price - entry_price
            if risk <= 0: continue
            tp_price = entry_price - (risk * tp_r)

        future = df_m3[df_m3['timestamp_ny'] > entry_time].head(120)
        res = 'TIMEOUT'
        
        for _, bar in future.iterrows():
            if direction == 'LONG':
                # Long exit at BID
                if bar['low_bid'] <= sl_price: res = 'SL'; break
                if bar['high_bid'] >= tp_price: res = 'TP'; break
            else:
                # Short exit at ASK (Bid + Spread)
                # sl_price and tp_price are compared against ASK
                ask_high = bar['high_bid'] + (spread_pips * 0.0001)
                ask_low = bar['low_bid'] + (spread_pips * 0.0001)
                if ask_high >= sl_price: res = 'SL'; break
                if ask_low <= tp_price: res = 'TP'; break
        
        results.append(res)
        
    res_s = pd.Series(results)
    tp_c = len(res_s[res_s == 'TP'])
    sl_c = len(res_s[res_s == 'SL'])
    pf = round((tp_c * tp_r) / sl_c, 2) if sl_c > 0 else 0
    
    print(f"Auditoría de Ejecución Finalizada. PF (Spread 0.3 + Slip 0.2): {pf}")
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase18_forensic_audit\execution")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "phase18_bid_ask_audit.json", 'w') as f:
        json.dump({"pf_strict": pf, "sample": len(res_s)}, f, indent=2)

if __name__ == "__main__":
    run_forensic_execution_audit()

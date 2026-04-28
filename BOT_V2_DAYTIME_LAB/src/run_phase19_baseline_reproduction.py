
import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector
import json

def run_baseline_reproduction():
    print("Fase 1: Reproducción Baseline Phase 18...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    
    period = "period_2020_2026"
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    
    # 1. Detect Sweeps
    sweep_detector = H1FractalSweepDetector({})
    sweeps = sweep_detector.detect_sweeps(df_h1)
    
    # 2. Detect CHoCH (M3 as per baseline)
    choch_detector = First3MChochDetector({'max_mins_post_sweep': 60, 'sl_buffer': 0.5})
    signals = choch_detector.detect_choch(df_m3, sweeps)
    
    # 3. Apply Filters (Body 0.7 + Window 08-11 + First-of-day)
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
    
    # First of day
    sig_df['date'] = sig_df['choch_time'].dt.date
    sig_df = sig_df.sort_values('choch_time').groupby('date').head(1)
    
    # 4. Backtest (TP 2R, 0.5 slip)
    tp_r = 2.0
    slippage_pips = 0.5
    trades = []
    
    for _, sig in sig_df.iterrows():
        entry_time = sig['choch_time']
        direction = sig['direction']
        entry_price = sig['entry_price']
        sl_price = sig['sl_price']
        
        # Apply costs
        if direction == 'LONG': entry_price += slippage_pips * 0.0001
        else: entry_price -= slippage_pips * 0.0001
        
        risk = abs(entry_price - sl_price)
        if risk <= 0.00001: continue
        tp_price = entry_price + (risk * tp_r) if direction == 'LONG' else entry_price - (risk * tp_r)
        
        future = df_m3[df_m3['timestamp_ny'] > entry_time].head(120) # 6 hours timeout
        res = 'TIMEOUT'
        
        for _, bar in future.iterrows():
            if direction == 'LONG':
                if bar['low_bid'] <= sl_price: res = 'SL'; break
                if bar['high_bid'] >= tp_price: res = 'TP'; break
            else:
                if bar['high_bid'] >= sl_price: res = 'SL'; break
                if bar['low_bid'] <= tp_price: res = 'TP'; break
        
        trades.append({
            "time": entry_time,
            "dir": direction,
            "res": res,
            "r_pnl": 2.0 if res == 'TP' else (-1.0 if res == 'SL' else 0.0)
        })
        
    t_df = pd.DataFrame(trades)
    
    # 5. Metrics
    tp_c = len(t_df[t_df['res'] == 'TP'])
    sl_c = len(t_df[t_df['res'] == 'SL'])
    pf = round((tp_c * 2.0) / sl_c, 2) if sl_c > 0 else 0
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase19_phase18_expansion\baseline_reproduction")
    out_dir.mkdir(parents=True, exist_ok=True)
    t_df.to_csv(out_dir / "phase18_baseline_reproduced_trades.csv", index=False)
    
    summary = {
        "sample": len(t_df),
        "pf": pf,
        "win_rate": round(tp_c / len(t_df) * 100, 2) if len(t_df) > 0 else 0,
        "verdict": "PHASE18_BASELINE_REPRODUCED" if pf == 1.63 else "PHASE18_BASELINE_MISMATCH"
    }
    
    with open(out_dir / "phase18_baseline_reproduced_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Reproduction Complete. PF: {pf}. Verdict: {summary['verdict']}")

if __name__ == "__main__":
    run_baseline_reproduction()


import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector
import json

def run_phase18_robustness_and_costs():
    print("Starting Phase 18 Robustness and Costs...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    
    period = "period_2020_2026"
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    
    # 1. Detect Best Candidate: Body 0.7, TP 2R
    sweep_detector = H1FractalSweepDetector({})
    sweeps = sweep_detector.detect_sweeps(df_h1)
    choch_detector = First3MChochDetector({'max_mins_post_sweep': 60, 'sl_buffer': 0.5})
    signals = choch_detector.detect_choch(df_m3, sweeps)
    
    # Pre-calculate body filter
    df_m3['body'] = (df_m3['close_bid'] - df_m3['open_bid']).abs()
    df_m3['range'] = df_m3['high_bid'] - df_m3['low_bid']
    df_m3['body_pct'] = df_m3['body'] / df_m3['range'].replace(0, 0.00001)
    
    # Map back and filter
    signals_filtered = []
    for idx, sig in signals.iterrows():
        m3_bar = df_m3[df_m3['timestamp_ny'] == sig['choch_time']].iloc[0]
        if m3_bar['body_pct'] >= 0.7:
            signals_filtered.append(sig)
    
    sig_df = pd.DataFrame(signals_filtered)
    sig_df['hour'] = sig_df['choch_time'].dt.hour
    sig_df = sig_df[(sig_df['hour'] >= 8) & (sig_df['hour'] <= 10)]
    
    # Apply "First Trade of the day" filter
    sig_df['date'] = sig_df['choch_time'].dt.date
    sig_df = sig_df.sort_values('choch_time').groupby('date').head(1)
    print(f"Sample after first-of-day filter: {len(sig_df)}")
    
    # Backtest with costs
    results = []
    tp_r = 2.0
    for slippage_pips in [0.0, 0.5, 1.0]:
        trades = []
        for _, sig in sig_df.iterrows():
            entry_time = sig['choch_time']
            direction = sig['direction']
            entry_price = sig['entry_price']
            sl_price = sig['sl_price']
            
            # Apply slippage
            if direction == 'LONG': entry_price += slippage_pips * 0.0001
            else: entry_price -= slippage_pips * 0.0001
            
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
            trades.append({"res": res, "year": entry_time.year})
            
        t_df = pd.DataFrame(trades)
        if t_df.empty: continue
        
        tp_count = len(t_df[t_df['res'] == 'TP'])
        sl_count = len(t_df[t_df['res'] == 'SL'])
        pf = round((tp_count * tp_r) / sl_count, 2) if sl_count > 0 else 0
        
        results.append({
            "slippage": slippage_pips,
            "sample": len(t_df),
            "pf": pf,
            "win_rate": round(tp_count / len(t_df) * 100, 2)
        })
        
        if slippage_pips == 0.5: # Save year by year for 0.5 slippage
            by_year = t_df.groupby('year').apply(lambda x: round((len(x[x['res']=='TP'])*tp_r)/len(x[x['res']=='SL']), 2) if len(x[x['res']=='SL'])>0 else 0)
            robust_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase18_h1_fractal_sweep\robustness")
            robust_dir.mkdir(parents=True, exist_ok=True)
            by_year.to_csv(robust_dir / "phase18_by_year.csv")

    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase18_h1_fractal_sweep\execution")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "phase18_cost_sensitivity.csv", index=False)
    print("Robustness and Costs Complete.")

if __name__ == "__main__":
    run_phase18_robustness_and_costs()

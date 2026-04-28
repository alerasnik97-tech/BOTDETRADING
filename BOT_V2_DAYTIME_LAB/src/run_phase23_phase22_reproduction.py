
import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector
from news_fortress.news_fortress_gate import NewsFortressGate
import os
import json

def run_reproduction_optimized():
    print("Fase 1 (Optimized): Reproducing Phase 22...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    period = "period_2020_2026"
    
    # Load HTF first for signals
    print("Loading H1 data...")
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    sweeps = H1FractalSweepDetector({}).detect_sweeps(df_h1)
    
    # Load M3 for signal detection
    print("Loading M3 data (Signal Detection)...")
    df_ltf = engine.load_and_prep_prices(period, timeframe='m3')
    choch_detector = First3MChochDetector({'max_mins_post_sweep': 60, 'sl_buffer': 0.5})
    signals = choch_detector.detect_choch(df_ltf, sweeps)
    
    # Body filter
    df_ltf['body'] = (df_ltf['close_bid'] - df_ltf['open_bid']).abs()
    df_ltf['range'] = df_ltf['high_bid'] - df_ltf['low_bid']
    df_ltf['body_pct'] = df_ltf['body'] / df_ltf['range'].replace(0, 0.00001)
    signals = pd.merge(signals, df_ltf[['timestamp_ny', 'body_pct']], left_on='choch_time', right_on='timestamp_ny', how='left')
    signals = signals[signals['body_pct'] >= 0.7].copy()
    signals['hour'] = signals['choch_time'].dt.hour
    signals = signals[(signals['hour'] >= 7) & (signals['hour'] < 16.5)]
    signals['date'] = signals['choch_time'].dt.date
    signals = signals.sort_values('choch_time').groupby('date').head(1)
    
    # News Gate
    print("Evaluating News Fortress...")
    df_news = engine.load_news(period)
    gate = NewsFortressGate(df_news)
    
    final_signals = []
    for _, sig in signals.iterrows():
        allow, _ = gate.evaluate_trading_permission(sig['choch_time'])
        if allow:
            final_signals.append(sig)
    
    signals = pd.DataFrame(final_signals)
    print(f"Final signals: {len(signals)}")
    
    # Execution (Optimized)
    print("Executing backtest...")
    df_ltf_indexed = df_ltf.set_index('timestamp_ny').sort_index()
    trades = []
    tp_r, be_r = 1.1, 0.5
    
    for _, sig in signals.iterrows():
        entry_time, direction, entry_p_orig, sl_p = sig['choch_time'], sig['direction'], sig['entry_price'], sig['sl_price']
        entry_price = entry_p_orig + 0.00005 if direction == 'LONG' else entry_p_orig - 0.00005
        risk = abs(entry_price - sl_p)
        if risk <= 0.00001: continue
        tp_price = entry_price + (risk * tp_r) if direction == 'LONG' else entry_price - (risk * tp_r)
        be_trigger = entry_price + (risk * be_r) if direction == 'LONG' else entry_price - (risk * be_r)
        
        # Only look at the next 4 hours (80 bars of 3m)
        future = df_ltf_indexed.loc[entry_time:].iloc[1:81]
        
        res, exit_time, exit_price = 'TIMEOUT', pd.NaT, np.nan
        curr_sl = sl_p
        for t, bar in future.iterrows():
            if direction == 'LONG':
                if bar['low_bid'] <= curr_sl: 
                    res = 'SL' if curr_sl == sl_p else 'BE'
                    exit_time, exit_price = t, curr_sl
                    break
                if bar['high_bid'] >= tp_price: 
                    res = 'TP'
                    exit_time, exit_price = t, tp_price
                    break
                if bar['high_bid'] >= be_trigger: 
                    curr_sl = entry_price
            else:
                if bar['high_ask'] >= curr_sl: 
                    res = 'SL' if curr_sl == sl_p else 'BE'
                    exit_time, exit_price = t, curr_sl
                    break
                if bar['low_bid'] <= tp_price: 
                    res = 'TP'
                    exit_time, exit_price = t, tp_price
                    break
                if bar['low_bid'] <= be_trigger: 
                    curr_sl = entry_price
        
        trades.append({
            "entry_time": entry_time, "direction": direction, "entry_price": entry_price, 
            "sl_price": sl_p, "tp_price": tp_price, "risk": risk, "res": res, 
            "exit_time": exit_time, "exit_price": exit_price, "pnl_r": 1.1 if res == 'TP' else (-1.0 if res == 'SL' else 0.0)
        })
        
    t_df = pd.DataFrame(trades)
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase23_phase22_forensic_readiness\reproduction")
    out_dir.mkdir(parents=True, exist_ok=True)
    t_df.to_csv(out_dir / "phase22_reproduced_trades_full.csv", index=False)
    print(f"File saved. Sample: {len(t_df)}")

if __name__ == "__main__":
    run_reproduction_optimized()

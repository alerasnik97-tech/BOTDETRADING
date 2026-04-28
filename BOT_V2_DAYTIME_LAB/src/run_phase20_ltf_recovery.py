
import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector
from news_fortress.news_fortress_gate import NewsFortressGate
import json

def run_ltf_recovery():
    print("Fase 4: Testing LTF Variants for Frequency Recovery...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    period = "period_2020_2026"
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_news = engine.load_news(period)
    gate = NewsFortressGate(df_news)
    
    sweep_detector = H1FractalSweepDetector({})
    sweeps = sweep_detector.detect_sweeps(df_h1)
    
    timeframes = ['m3', 'm5']
    bodies = [0.5, 0.6, 0.7]
    
    results = []
    
    for tf in timeframes:
        print(f"Testing TF: {tf}")
        df_ltf = engine.load_and_prep_prices(period, timeframe=tf)
        df_ltf['body'] = (df_ltf[f'close_bid'] - df_ltf[f'open_bid']).abs()
        df_ltf['range'] = df_ltf[f'high_bid'] - df_ltf[f'low_bid']
        df_ltf['body_pct'] = df_ltf['body'] / df_ltf['range'].replace(0, 0.00001)
        
        choch_detector = First3MChochDetector({'max_mins_post_sweep': 60, 'sl_buffer': 0.5})
        signals_all = choch_detector.detect_choch(df_ltf, sweeps)
        signals_all = pd.merge(signals_all, df_ltf[['timestamp_ny', 'body_pct']], left_on='choch_time', right_on='timestamp_ny', how='left')
        
        df_ltf_indexed = df_ltf.set_index('timestamp_ny').sort_index()
        
        for b_pct in bodies:
            label = f"{tf}_body{int(b_pct*100)}"
            sig_b = signals_all[signals_all['body_pct'] >= b_pct].copy()
            sig_b['hour'] = sig_b['choch_time'].dt.hour
            # Window 07:00 - 16:30 (The best from Phase 3)
            sig_b = sig_b[(sig_b['hour'] >= 7) & (sig_b['hour'] < 16.5)]
            sig_b['date'] = sig_b['choch_time'].dt.date
            sig_b = sig_b.sort_values('choch_time').groupby('date').head(1)
            
            trades = []
            for _, sig in sig_b.iterrows():
                allow, _ = gate.evaluate_trading_permission(sig['choch_time'])
                if not allow: continue
                
                entry_time = sig['choch_time']
                direction = sig['direction']
                entry_price = sig['entry_price']
                sl_price = sig['sl_price']
                
                if direction == 'LONG': entry_price += 0.00005
                else: entry_price -= 0.00005
                
                risk = abs(entry_price - sl_price)
                if risk <= 0.00001: continue
                tp_price = entry_price + (risk * 2.0) if direction == 'LONG' else entry_price - (risk * 2.0)
                
                try:
                    future = df_ltf_indexed.loc[entry_time:].iloc[1:121]
                except: continue
                
                res = 'TIMEOUT'
                for _, bar in future.iterrows():
                    if direction == 'LONG':
                        if bar['low_bid'] <= sl_price: res = 'SL'; break
                        if bar['high_bid'] >= tp_price: res = 'TP'; break
                    else:
                        if bar['high_ask'] >= sl_price: res = 'SL'; break
                        if bar['low_bid'] <= tp_price: res = 'TP'; break
                trades.append(res)
                
            t_s = pd.Series(trades)
            tp_c, sl_c = len(t_s[t_s == 'TP']), len(t_s[t_s == 'SL'])
            pf = round((tp_c * 2.0) / sl_c, 2) if sl_c > 0 else 0
            results.append({"variant": label, "sample": len(t_s), "pf": pf})
            print(f"Variant {label}: PF {pf} | Sample {len(t_s)}")
            
    pd.DataFrame(results).to_csv(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase20_news_fortress_frequency_recovery\ltf_recovery\ltf_recovery_results.csv", index=False)

if __name__ == "__main__":
    run_ltf_recovery()

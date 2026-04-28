
import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector
from news_fortress.news_fortress_gate import NewsFortressGate
import json

def run_window_recovery():
    print("Fase 3: Testing Window Expansion for Frequency Recovery...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    period = "period_2020_2026"
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    df_news = engine.load_news(period)
    
    gate = NewsFortressGate(df_news)
    sweep_detector = H1FractalSweepDetector({})
    sweeps = sweep_detector.detect_sweeps(df_h1)
    choch_detector = First3MChochDetector({'max_mins_post_sweep': 60, 'sl_buffer': 0.5})
    signals = choch_detector.detect_choch(df_m3, sweeps)
    
    df_m3['body'] = (df_m3['close_bid'] - df_m3['open_bid']).abs()
    df_m3['range'] = df_m3['high_bid'] - df_m3['low_bid']
    df_m3['body_pct'] = df_m3['body'] / df_m3['range'].replace(0, 0.00001)
    signals = pd.merge(signals, df_m3[['timestamp_ny', 'body_pct']], left_on='choch_time', right_on='timestamp_ny', how='left')
    signals = signals[signals['body_pct'] >= 0.7].copy()
    signals['hour'] = signals['choch_time'].dt.hour
    
    windows = [
        (8, 11), (8, 14), (8, 16.5), (7, 16.5), (7, 20)
    ]
    
    results = []
    df_m3_indexed = df_m3.set_index('timestamp_ny').sort_index()
    
    for start_h, end_h in windows:
        win_label = f"{start_h}:00-{end_h}:00"
        sig_w = signals[(signals['hour'] >= start_h) & (signals['hour'] < end_h)].copy()
        sig_w['date'] = sig_w['choch_time'].dt.date
        # Max 1 trade per day for fair comparison
        sig_w = sig_w.sort_values('choch_time').groupby('date').head(1)
        
        trades = []
        for _, sig in sig_w.iterrows():
            # News Fortress Check
            allow, reason = gate.evaluate_trading_permission(sig['choch_time'])
            if not allow: continue
            
            entry_time = sig['choch_time']
            direction = sig['direction']
            entry_price = sig['entry_price']
            sl_price = sig['sl_price']
            
            # Costs
            if direction == 'LONG': entry_price += 0.00005
            else: entry_price -= 0.00005
            
            risk = abs(entry_price - sl_price)
            if risk <= 0.00001: continue
            tp_price = entry_price + (risk * 2.0) if direction == 'LONG' else entry_price - (risk * 2.0)
            
            try:
                future = df_m3_indexed.loc[entry_time:].iloc[1:121]
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
        results.append({"window": win_label, "sample": len(t_s), "pf": pf})
        print(f"Window {win_label}: PF {pf} | Sample {len(t_s)}")
        
    pd.DataFrame(results).to_csv(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase20_news_fortress_frequency_recovery\window_recovery\window_recovery_results.csv", index=False)

if __name__ == "__main__":
    run_window_recovery()

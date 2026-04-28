
import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector
from news_fortress.news_fortress_gate import NewsFortressGate
import json

def run_profiles():
    print("Fase 3: Testing Management Profiles for Phase 20...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    period = "period_2020_2026"
    df_ltf = engine.load_and_prep_prices(period, timeframe='m3')
    df_news = engine.load_news(period)
    gate = NewsFortressGate(df_news)
    
    # Generate signals once
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    sweeps = H1FractalSweepDetector({}).detect_sweeps(df_h1)
    signals = First3MChochDetector({'max_mins_post_sweep': 60, 'sl_buffer': 0.5}).detect_choch(df_ltf, sweeps)
    df_ltf['body'] = (df_ltf['close_bid'] - df_ltf['open_bid']).abs()
    df_ltf['range'] = df_ltf['high_bid'] - df_ltf['low_bid']
    df_ltf['body_pct'] = df_ltf['body'] / df_ltf['range'].replace(0, 0.00001)
    signals = pd.merge(signals, df_ltf[['timestamp_ny', 'body_pct']], left_on='choch_time', right_on='timestamp_ny', how='left')
    signals = signals[signals['body_pct'] >= 0.7].copy()
    signals['hour'] = signals['choch_time'].dt.hour
    signals = signals[(signals['hour'] >= 7) & (signals['hour'] < 16.5)]
    signals['date'] = signals['choch_time'].dt.date
    signals = signals.sort_values('choch_time').groupby('date').head(1)
    
    df_ltf_indexed = df_ltf.set_index('timestamp_ny').sort_index()
    
    profiles = [
        {"name": "Conservative", "tp": 1.25, "be": 0.75},
        {"name": "Balanced", "tp": 1.5, "be": 1.0},
        {"name": "Institutional", "tp": 2.0, "be": 1.0}, # The current best
        {"name": "Aggressive", "tp": 2.5, "be": None}
    ]
    
    results = []
    
    for p in profiles:
        print(f"Testing Profile: {p['name']}")
        trades = []
        for _, sig in signals.iterrows():
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
            tp_price = entry_price + (risk * p['tp']) if direction == 'LONG' else entry_price - (risk * p['tp'])
            
            try:
                future = df_ltf_indexed.loc[entry_time:].iloc[1:121]
            except: continue
            
            res = 'TIMEOUT'
            curr_sl = sl_price
            for _, bar in future.iterrows():
                if direction == 'LONG':
                    if bar['low_bid'] <= curr_sl: res = 'SL' if curr_sl == sl_price else 'BE'; break
                    if bar['high_bid'] >= tp_price: res = 'TP'; break
                    if p['be'] and bar['high_bid'] >= entry_price + (risk * p['be']): curr_sl = entry_price
                else:
                    if bar['high_ask'] >= curr_sl: res = 'SL' if curr_sl == sl_price else 'BE'; break
                    if bar['low_bid'] <= tp_price: res = 'TP'; break
                    if p['be'] and bar['low_bid'] <= entry_price - (risk * p['be']): curr_sl = entry_price
            
            pnl = p['tp'] if res == 'TP' else (-1.0 if res == 'SL' else 0.0)
            trades.append(pnl)
            
        t_s = pd.Series(trades)
        tp_c = len(t_s[t_s > 0])
        sl_c = len(t_s[t_s < 0])
        pf = round((tp_c * p['tp']) / sl_c, 2) if sl_c > 0 else 0
        wr = round(tp_c / len(t_s), 3) if len(t_s) > 0 else 0
        
        # Streak calculation
        is_loss = (t_s < 0).astype(int)
        max_streak = (is_loss * (is_loss.groupby((is_loss != is_loss.shift()).cumsum()).cumcount() + 1)).max()
        
        results.append({
            "profile": p['name'], "tp": p['tp'], "be": p['be'],
            "sample": len(t_s), "pf": pf, "winrate": wr, "max_streak": int(max_streak)
        })
        print(f"{p['name']}: PF {pf} | WR {wr} | Streak {max_streak}")
        
    pd.DataFrame(results).to_csv(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase21_operability_decision\phase20_management_profiles\phase20_management_profiles.csv", index=False)

if __name__ == "__main__":
    run_profiles()

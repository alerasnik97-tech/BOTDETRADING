
import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector
from news_fortress.news_fortress_gate import NewsFortressGate
import json

def run_management_matrix():
    print("Fase 2: High Winrate Management Matrix...")
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
    
    tps = [0.75, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 1.75]
    bes = [None, 0.5, 0.75, 1.0]
    
    results = []
    
    for tp_r in tps:
        for be_r in bes:
            label = f"TP{tp_r}_BE{be_r}"
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
                tp_price = entry_price + (risk * tp_r) if direction == 'LONG' else entry_price - (risk * tp_r)
                
                try:
                    future = df_ltf_indexed.loc[entry_time:].iloc[1:121]
                except: continue
                
                res = 'TIMEOUT'
                curr_sl = sl_price
                for _, bar in future.iterrows():
                    if direction == 'LONG':
                        if bar['low_bid'] <= curr_sl: res = 'SL' if curr_sl == sl_price else 'BE'; break
                        if bar['high_bid'] >= tp_price: res = 'TP'; break
                        if be_r and bar['high_bid'] >= entry_price + (risk * be_r): curr_sl = entry_price
                    else:
                        if bar['high_ask'] >= curr_sl: res = 'SL' if curr_sl == sl_price else 'BE'; break
                        if bar['low_bid'] <= tp_price: res = 'TP'; break
                        if be_r and bar['low_bid'] <= entry_price - (risk * be_r): curr_sl = entry_price
                trades.append(res)
                
            t_s = pd.Series(trades)
            tp_c, sl_c = len(t_s[t_s == 'TP']), len(t_s[t_s == 'SL'])
            pf = round((tp_c * tp_r) / sl_c, 2) if sl_c > 0 else 0
            wr = round(tp_c / len(t_s), 3) if len(t_s) > 0 else 0
            
            # Streak and DD
            pnl_r = [tp_r if r == 'TP' else (-1.0 if r == 'SL' else 0.0) for r in trades]
            cum_pnl = np.cumsum(pnl_r)
            dd = cum_pnl - np.maximum.accumulate(cum_pnl)
            max_dd = round(np.min(dd), 2) if len(dd) > 0 else 0
            
            is_loss = (pd.Series(pnl_r) < 0).astype(int)
            max_streak = (is_loss * (is_loss.groupby((is_loss != is_loss.shift()).cumsum()).cumcount() + 1)).max()
            
            results.append({
                "config": label, "sample": len(t_s), "pf": pf, "winrate": wr, "max_dd": max_dd, "max_streak": int(max_streak)
            })
            print(f"Config {label}: PF {pf} | WR {wr} | DD {max_dd}")
            
    pd.DataFrame(results).to_csv(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase22_high_wr_low_dd\management_matrix\high_wr_management_results.csv", index=False)

if __name__ == "__main__":
    run_management_matrix()

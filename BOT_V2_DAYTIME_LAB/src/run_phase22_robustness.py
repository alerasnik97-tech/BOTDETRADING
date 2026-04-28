
import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector
from news_fortress.news_fortress_gate import NewsFortressGate
import json

def run_robustness():
    print("Fase 7: Robustness Analysis for Winner Profile...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    period = "period_2020_2026"
    df_ltf = engine.load_and_prep_prices(period, timeframe='m3')
    df_news = engine.load_news(period)
    gate = NewsFortressGate(df_news)
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    sweeps = H1FractalSweepDetector({}).detect_sweeps(df_h1)
    
    choch_detector = First3MChochDetector({'max_mins_post_sweep': 60, 'sl_buffer': 0.5})
    signals_all = choch_detector.detect_choch(df_ltf, sweeps)
    df_ltf['body'] = (df_ltf['close_bid'] - df_ltf['open_bid']).abs()
    df_ltf['range'] = df_ltf['high_bid'] - df_ltf['low_bid']
    df_ltf['body_pct'] = df_ltf['body'] / df_ltf['range'].replace(0, 0.00001)
    signals_all = pd.merge(signals_all, df_ltf[['timestamp_ny', 'body_pct']], left_on='choch_time', right_on='timestamp_ny', how='left')
    signals_all = signals_all[signals_all['body_pct'] >= 0.7].copy()
    signals_all['hour'] = signals_all['choch_time'].dt.hour
    signals_all = signals_all[(signals_all['hour'] >= 7) & (signals_all['hour'] < 16.5)]
    signals_all['date'] = signals_all['choch_time'].dt.date
    signals_all = signals_all.sort_values('choch_time').groupby('date').head(1)
    
    df_ltf_indexed = df_ltf.set_index('timestamp_ny').sort_index()
    tp_r = 1.1
    be_r = 0.5
    
    trades = []
    for _, sig in signals_all.iterrows():
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
        trades.append({"time": entry_time, "pnl_r": 1.1 if res == 'TP' else (-1.0 if res == 'SL' else 0.0)})
        
    t_df = pd.DataFrame(trades)
    t_df['year'] = t_df['time'].dt.year
    robustness = t_df.groupby('year')['pnl_r'].agg([
        ('sample', 'count'),
        ('pf', lambda x: round((x[x > 0].sum()) / abs(x[x < 0].sum()), 2) if abs(x[x < 0].sum()) > 0 else 0),
        ('wr', lambda x: round((x > 0).mean(), 3))
    ]).reset_index()
    
    print(robustness)
    robustness.to_csv(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase22_high_wr_low_dd\robustness\phase22_robustness_by_year.csv", index=False)

if __name__ == "__main__":
    run_robustness()

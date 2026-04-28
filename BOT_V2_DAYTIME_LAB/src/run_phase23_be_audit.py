
import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector
from news_fortress.news_fortress_gate import NewsFortressGate
import json

def run_be_audit_full():
    print("Fase 2 (Full): Conservative BE 0.5R Audit with prices...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    period = "period_2020_2026"
    df_ltf = engine.load_and_prep_prices(period, timeframe='m3')
    df_news = engine.load_news(period)
    gate = NewsFortressGate(df_news)
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    sweeps = H1FractalSweepDetector({}).detect_sweeps(df_h1)
    choch_detector = First3MChochDetector({'max_mins_post_sweep': 60, 'sl_buffer': 0.5})
    signals = choch_detector.detect_choch(df_ltf, sweeps)
    
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
    tp_r, be_r = 1.1, 0.5
    audit_trades = []
    for _, sig in signals.iterrows():
        allow, _ = gate.evaluate_trading_permission(sig['choch_time'])
        if not allow: continue
        entry_time, direction, entry_p_orig, sl_p = sig['choch_time'], sig['direction'], sig['entry_price'], sig['sl_price']
        
        # Sincronización con Phase14Engine: LONG usa Ask, SHORT usa Bid
        try:
            candle = df_ltf_indexed.loc[entry_time]
            real_entry_p = candle['close_ask'] if direction == 'LONG' else candle['close_bid']
        except:
            real_entry_p = entry_p_orig
            
        entry_price = real_entry_p + 0.00005 if direction == 'LONG' else real_entry_p - 0.00005
        risk = abs(entry_price - sl_p)
        if risk <= 0.00001: continue
        tp_price = entry_price + (risk * tp_r) if direction == 'LONG' else entry_price - (risk * tp_r)
        be_trigger = entry_price + (risk * be_r) if direction == 'LONG' else entry_price - (risk * be_r)
        
        try:
            future = df_ltf_indexed.loc[entry_time:].iloc[1:81]
        except: continue
        
        res, exit_time, exit_price = 'TIMEOUT', pd.NaT, np.nan
        curr_sl = sl_p
        is_be_active = False
        ambiguous = False
        for t, bar in future.iterrows():
            if direction == 'LONG':
                can_trig_be = bar['high_bid'] >= be_trigger
                can_hit_sl = bar['low_bid'] <= curr_sl
                if can_trig_be and can_hit_sl and not is_be_active:
                    res, exit_time, exit_price, ambiguous = 'SL', t, curr_sl, True
                    break
                if can_hit_sl:
                    res, exit_time, exit_price = ('SL' if not is_be_active else 'BE'), t, curr_sl
                    break
                if bar['high_bid'] >= tp_price:
                    res, exit_time, exit_price = 'TP', t, tp_price
                    break
                if can_trig_be:
                    is_be_active, curr_sl = True, entry_price
            else:
                can_trig_be = bar['low_bid'] <= be_trigger
                can_hit_sl = bar['high_ask'] >= curr_sl
                if can_trig_be and can_hit_sl and not is_be_active:
                    res, exit_time, exit_price, ambiguous = 'SL', t, curr_sl, True
                    break
                if can_hit_sl:
                    res, exit_time, exit_price = ('SL' if not is_be_active else 'BE'), t, curr_sl
                    break
                if bar['low_bid'] <= tp_price:
                    res, exit_time, exit_price = 'TP', t, tp_price
                    break
                if can_trig_be:
                    is_be_active, curr_sl = True, entry_price
                    
        audit_trades.append({
            "entry_time": entry_time, "direction": direction, "entry_price": entry_price, 
            "sl_price": sl_p, "tp_price": tp_price, "risk": risk, "res": res, 
            "exit_time": exit_time, "exit_price": exit_price, "ambiguous": ambiguous,
            "pnl_r": 1.1 if res == 'TP' else (-1.0 if res == 'SL' else 0.0)
        })
        
    a_df = pd.DataFrame(audit_trades)
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase23_phase22_forensic_readiness\be_audit")
    a_df.to_csv(out_dir / "phase22_be_05_audit_full.csv", index=False)
    print(f"Full BE Audit complete. Sample: {len(a_df)}")

if __name__ == "__main__":
    run_be_audit_full()

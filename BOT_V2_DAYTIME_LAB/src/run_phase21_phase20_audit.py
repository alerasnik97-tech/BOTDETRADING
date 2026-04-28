
import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector
from news_fortress.news_fortress_gate import NewsFortressGate
import json

def run_audit():
    print("Fase 2: Phase 20 Operability Audit...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    period = "period_2020_2026"
    df_ltf = engine.load_and_prep_prices(period, timeframe='m3')
    df_news = engine.load_news(period)
    gate = NewsFortressGate(df_news)
    
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    sweep_detector = H1FractalSweepDetector({})
    sweeps = sweep_detector.detect_sweeps(df_h1)
    choch_detector = First3MChochDetector({'max_mins_post_sweep': 60, 'sl_buffer': 0.5})
    signals = choch_detector.detect_choch(df_ltf, sweeps)
    
    df_ltf['body'] = (df_ltf['close_bid'] - df_ltf['open_bid']).abs()
    df_ltf['range'] = df_ltf['high_bid'] - df_ltf['low_bid']
    df_ltf['body_pct'] = df_ltf['body'] / df_ltf['range'].replace(0, 0.00001)
    signals = pd.merge(signals, df_ltf[['timestamp_ny', 'body_pct']], left_on='choch_time', right_on='timestamp_ny', how='left')
    
    # Phase 20 best config
    signals = signals[signals['body_pct'] >= 0.7].copy()
    signals['hour'] = signals['choch_time'].dt.hour
    signals = signals[(signals['hour'] >= 7) & (signals['hour'] < 16.5)]
    signals['date'] = signals['choch_time'].dt.date
    signals = signals.sort_values('choch_time').groupby('date').head(1)
    
    df_ltf_indexed = df_ltf.set_index('timestamp_ny').sort_index()
    
    trades = []
    tp_r = 2.0
    be_r = 1.0
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
        
        trades.append({
            "time": entry_time,
            "res": res,
            "pnl_r": 2.0 if res == 'TP' else (-1.0 if res == 'SL' else 0.0)
        })
        
    t_df = pd.DataFrame(trades)
    
    # Audit Metrics
    t_df['cum_pnl'] = t_df['pnl_r'].cumsum()
    t_df['peak'] = t_df['cum_pnl'].cummax()
    t_df['drawdown'] = t_df['cum_pnl'] - t_df['peak']
    
    max_dd = t_df['drawdown'].min()
    
    # Loss streaks
    t_df['is_loss'] = (t_df['pnl_r'] < 0).astype(int)
    t_df['streak'] = t_df['is_loss'] * (t_df['is_loss'].groupby((t_df['is_loss'] != t_df['is_loss'].shift()).cumsum()).cumcount() + 1)
    max_streak = t_df['streak'].max()
    
    tp_c, sl_c, be_c = len(t_df[t_df['res'] == 'TP']), len(t_df[t_df['res'] == 'SL']), len(t_df[t_df['res'] == 'BE'])
    pf = round((tp_c * 2.0) / sl_c, 2) if sl_c > 0 else 0
    
    summary = {
        "sample": len(t_df),
        "pf": pf,
        "winrate": round(tp_c / len(t_df), 3),
        "max_drawdown_r": round(max_dd, 2),
        "max_loss_streak": int(max_streak),
        "tp_count": tp_c,
        "sl_count": sl_c,
        "be_count": be_c
    }
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase21_operability_decision\phase20_operability")
    t_df.to_csv(out_dir / "phase20_operability_metrics.csv", index=False)
    with open(out_dir / "phase20_operability_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Audit Complete. PF: {pf} | Max DD: {max_dd}R | Max Streak: {max_streak}")

if __name__ == "__main__":
    run_audit()

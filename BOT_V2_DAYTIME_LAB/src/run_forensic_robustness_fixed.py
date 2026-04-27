
import pandas as pd
import numpy as np
import json
from pathlib import Path

def run_fixed_backtest():
    print("FASE 7: ROBUSTEZ TEMPORAL (LÓGICA CORREGIDA)")
    
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\ARCHIVE_SUPERSEDED\duplicated_folders\Bot V1_PENDING_DELETE\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    periods = ['period_2015_2019', 'period_2020_2026']
    
    # Load H1 for EMA 50
    h1_list = []
    for p in periods:
        df = pd.read_csv(manifest[p]['h1_bid'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('America/New_York')
        h1_list.append(df)
    df_h1 = pd.concat(h1_list).sort_values('timestamp')
    df_h1['ema50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
    
    # Load M5
    m5_list = []
    for p in periods:
        df = pd.read_csv(manifest[p]['m5_bid'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('America/New_York')
        m5_list.append(df)
    df_m5 = pd.concat(m5_list).sort_values('timestamp').reset_index(drop=True)
    df_m5['date'] = df_m5['timestamp'].dt.date
    
    # OR Range (08:00 - 08:30)
    df_m5['hour'] = df_m5['timestamp'].dt.hour
    df_m5['minute'] = df_m5['timestamp'].dt.minute
    or_range = df_m5[(df_m5['hour'] == 8) & (df_m5['minute'] < 30)].groupby('date').agg({'high': 'max', 'low': 'min'}).to_dict('index')
    
    # Sync EMA 50
    df_h1_sync = df_h1[['timestamp', 'ema50']].rename(columns={'timestamp': 'h1_time'})
    df_m5 = pd.merge_asof(df_m5.sort_values('timestamp'), df_h1_sync.sort_values('h1_time'), 
                         left_on='timestamp', right_on='h1_time', direction='backward')
    
    rows = list(df_m5.itertuples())
    trades = []
    tp_r = 2.0
    spread = 0.00007
    
    last_trade_date = None
    for i in range(1, len(rows)):
        row = rows[i]
        if row.date == last_trade_date: continue
        if row.timestamp.hour < 9 or row.timestamp.hour >= 13: continue
        
        lvl = or_range.get(row.date)
        if not lvl: continue
        prev = rows[i-1]
        dist = (row.close - row.ema50) * 10000
        
        signal = 0
        if prev.high > lvl['high'] and row.close < lvl['high'] and dist > 20: signal = -1; extreme = prev.high
        elif prev.low < lvl['low'] and row.close > lvl['low'] and dist < -20: signal = 1; extreme = prev.low
            
        if signal != 0:
            # FIX: Correct entry Bid/Ask and TP sign
            if signal == 1: # LONG
                entry_p = row.close + spread # Entry at ASK
                sl = extreme - 0.0001
                risk = abs(entry_p - sl)
                tp_p = entry_p + (risk * tp_r) # CORRECT SIGN
            else: # SHORT
                entry_p = row.close # Entry at BID
                sl = extreme + 0.0001
                risk = abs(entry_p - sl)
                tp_p = entry_p - (risk * tp_r) # CORRECT SIGN
                
            r_val = -1.0
            for j in range(i+1, min(i+300, len(rows))):
                f = rows[j]
                if signal == 1: # LONG (Exit at BID)
                    if f.low <= sl: r_val = -1.0; break
                    if f.high >= tp_p: r_val = tp_r; break
                else: # SHORT (Exit at ASK)
                    if (f.high + spread) >= sl: r_val = -1.0; break
                    if (f.low + spread) <= tp_p: r_val = tp_r; break
            
            trades.append({'date': row.date, 'r_val': r_val})
            last_trade_date = row.date
            
    df_trades = pd.DataFrame(trades)
    
    # Yearly Robustness
    df_trades['year'] = pd.to_datetime(df_trades['date']).dt.year
    yearly = df_trades.groupby('year')['r_val'].agg(['count', 'sum', 'mean'])
    yearly['pf'] = df_trades.groupby('year').apply(lambda x: x[x['r_val'] > 0]['r_val'].sum() / abs(x[x['r_val'] < 0]['r_val'].sum()) if len(x[x['r_val'] < 0]) > 0 else 99)
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase12_forensic_audit\robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(out_dir / "phase12_robustness_by_year.csv")
    
    # Summary
    gp = df_trades[df_trades['r_val'] > 0]['r_val'].sum()
    gl = abs(df_trades[df_trades['r_val'] < 0]['r_val'].sum())
    pf = gp / gl if gl > 0 else 0
    print(f"Fixed Logic Backtest Complete. TRUE PF: {pf:.3f}, Sample: {len(df_trades)}")

if __name__ == "__main__":
    run_fixed_backtest()

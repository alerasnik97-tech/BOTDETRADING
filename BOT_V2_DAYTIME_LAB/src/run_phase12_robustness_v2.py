
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta

def run_robustness():
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\ARCHIVE_SUPERSEDED\duplicated_folders\Bot V1_PENDING_DELETE\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    periods = ['period_2015_2019', 'period_2020_2026']
    
    h1_list = []
    for p in periods:
        df = pd.read_csv(manifest[p]['h1_bid'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('America/New_York')
        h1_list.append(df)
    df_h1 = pd.concat(h1_list).sort_values('timestamp')
    df_h1['ema50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
    
    m5_list = []
    for p in periods:
        df = pd.read_csv(manifest[p]['m5_bid'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('America/New_York')
        m5_list.append(df)
    df_m5 = pd.concat(m5_list).sort_values('timestamp').reset_index(drop=True)
    df_m5['date'] = df_m5['timestamp'].dt.date
    
    df_h1_sync = df_h1[['timestamp', 'ema50']].rename(columns={'timestamp': 'h1_time'})
    df_m5 = pd.merge_asof(df_m5.sort_values('timestamp'), df_h1_sync.sort_values('h1_time'), 
                         left_on='timestamp', right_on='h1_time', direction='backward')
    
    rows = list(df_m5.itertuples())
    robustness_results = []
    
    tp_test = 2.0
    
    # 1. Spread Stress
    for spread_pips in [0.7, 1.0, 1.5]:
        print(f"Testing Spread {spread_pips} pips...")
        trades = []
        last_trade_date = None
        spread = spread_pips * 0.0001
        
        # Calculate OR base
        or_range = df_m5[(df_m5['timestamp'].dt.hour == 8) & (df_m5['timestamp'].dt.minute < 30)].groupby('date').agg({'high': 'max', 'low': 'min'}).to_dict('index')
        
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
                entry_p = row.close
                sl = extreme + (0.0001 if signal == -1 else -0.0001)
                risk = abs(entry_p - sl)
                tp_p = entry_p + (risk * tp_test * signal * -1)
                r_val = -1.0
                for j in range(i+1, min(i+150, len(rows))):
                    f = rows[j]
                    if signal == 1:
                        if f.low <= sl: r_val = -1.0; break
                        if f.high >= tp_p: r_val = tp_test; break
                    else:
                        if (f.high + spread) >= sl: r_val = -1.0; break
                        if (f.low + spread) <= tp_p: r_val = tp_test; break
                trades.append(r_val)
                last_trade_date = row.date
        
        if trades:
            df_t = pd.Series(trades)
            pf = df_t[df_t > 0].sum() / abs(df_t[df_t < 0].sum()) if len(df_t[df_t < 0]) > 0 else 99
            robustness_results.append({"test": "Spread", "value": spread_pips, "pf": round(pf, 2)})

    # 2. Window Shift
    for shift in [-15, 0, 15]:
        print(f"Testing Window Shift {shift} mins...")
        trades = []
        last_trade_date = None
        spread = 0.00007
        or_range = df_m5[(df_m5['timestamp'].dt.hour == 8) & (df_m5['timestamp'].dt.minute < 30)].groupby('date').agg({'high': 'max', 'low': 'min'}).to_dict('index')
        
        for i in range(1, len(rows)):
            row = rows[i]
            if row.date == last_trade_date: continue
            
            # Shifting 9:00 - 13:00 window
            start_m = 9 * 60 + shift
            end_m = 13 * 60 + shift
            curr_m = row.timestamp.hour * 60 + row.timestamp.minute
            
            if curr_m < start_m or curr_m >= end_m: continue
            
            lvl = or_range.get(row.date)
            if not lvl: continue
            prev = rows[i-1]
            dist = (row.close - row.ema50) * 10000
            
            signal = 0
            if prev.high > lvl['high'] and row.close < lvl['high'] and dist > 20: signal = -1; extreme = prev.high
            elif prev.low < lvl['low'] and row.close > lvl['low'] and dist < -20: signal = 1; extreme = prev.low
            
            if signal != 0:
                entry_p = row.close
                sl = extreme + (0.0001 if signal == -1 else -0.0001)
                risk = abs(entry_p - sl)
                tp_p = entry_p + (risk * tp_test * signal * -1)
                r_val = -1.0
                for j in range(i+1, min(i+150, len(rows))):
                    f = rows[j]
                    if signal == 1:
                        if f.low <= sl: r_val = -1.0; break
                        if f.high >= tp_p: r_val = tp_test; break
                    else:
                        if (f.high + spread) >= sl: r_val = -1.0; break
                        if (f.low + spread) <= tp_p: r_val = tp_test; break
                trades.append(r_val)
                last_trade_date = row.date
        
        if trades:
            df_t = pd.Series(trades)
            pf = df_t[df_t > 0].sum() / abs(df_t[df_t < 0].sum()) if len(df_t[df_t < 0]) > 0 else 99
            robustness_results.append({"test": "TimeShift", "value": shift, "pf": round(pf, 2)})

    output_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase12_surpass_manual_pf\robustness")
    pd.DataFrame(robustness_results).to_csv(output_dir / "selective_fakeout_robustness.csv", index=False)
    print("Robustness Complete.")

if __name__ == "__main__":
    run_robustness()

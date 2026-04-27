
import pandas as pd
import numpy as np
import json
from pathlib import Path

class Method2V2Engine:
    def __init__(self):
        self.tz_ny = 'America/New_York'

    def run_screening(self, df_m5):
        print("Method 2 V2: Session Range Fade")
        trades = []
        df_m5 = df_m5.copy()
        df_m5['timestamp_ny'] = df_m5['timestamp'].dt.tz_convert(self.tz_ny)
        df_m5['date'] = df_m5['timestamp_ny'].dt.date
        
        # Asia Range (00:00 - 07:00)
        asia = df_m5[(df_m5['timestamp_ny'].dt.hour >= 0) & (df_m5['timestamp_ny'].dt.hour < 7)]
        asia_levels = asia.groupby('date').agg({'high': 'max', 'low': 'min'}).to_dict('index')
        
        i = 100
        total = len(df_m5)
        while i < total:
            row = df_m5.iloc[i]
            curr_time = row.timestamp_ny
            if curr_time.hour < 8 or curr_time.hour >= 12: i += 1; continue
            
            lvl = asia_levels.get(curr_time.date())
            if lvl:
                prev = df_m5.iloc[i-1]
                # Fakeout of Asia Range
                if prev.high > lvl['high'] and row.close < lvl['high']: # Failed upside
                    setup = {'direction': 'SHORT', 'entry_p': row.close, 'sl': prev.high + 0.0001, 'tp': lvl['low']}
                    res = self.resolve(df_m5, i, setup)
                    if res is not None: trades.append(res); i += 20; continue
                elif prev.low < lvl['low'] and row.close > lvl['low']: # Failed downside
                    setup = {'direction': 'LONG', 'entry_p': row.close, 'sl': prev.low - 0.0001, 'tp': lvl['high']}
                    res = self.resolve(df_m5, i, setup)
                    if res is not None: trades.append(res); i += 20; continue
            i += 1
        return pd.Series(trades)

    def resolve(self, df, idx, setup):
        sl, tp = setup['sl'], setup['tp']
        for j in range(idx + 1, min(idx + 200, len(df))):
            f = df.iloc[j]
            if setup['direction'] == 'LONG':
                if f.low <= sl: return -1.0
                if f.high >= tp: return (tp - setup['entry_p']) / (setup['entry_p'] - sl)
            else:
                if f.high >= sl: return -1.0
                if f.low <= tp: return (setup['entry_p'] - tp) / (sl - setup['entry_p'])
        return 0.0

if __name__ == "__main__":
    eng = Method2V2Engine()
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    p = 'period_2020_2026'
    df_m5 = pd.read_csv(manifest[p]['m5_bid'])
    df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'], utc=True)
    res = eng.run_screening(df_m5)
    if not res.empty:
        pf = res[res > 0].sum() / abs(res[res < 0].sum())
        print(f"Result: Sample={len(res)} PF={pf:.3f} Exp={res.mean():.4f}")



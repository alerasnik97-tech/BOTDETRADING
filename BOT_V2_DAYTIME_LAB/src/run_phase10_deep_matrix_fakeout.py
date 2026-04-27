
import pandas as pd
import numpy as np
import json
from pathlib import Path

class Phase10DeepMatrix:
    def __init__(self):
        self.tz_ny = 'America/New_York'

    def run_variant(self, df_m5, df_h1, dist_pips, tp_r):
        trades = []
        df_m5 = df_m5.copy()
        df_m5['timestamp_ny'] = df_m5['timestamp'].dt.tz_convert(self.tz_ny)
        df_h1['timestamp_ny'] = df_h1['timestamp'].dt.tz_convert(self.tz_ny)
        df_m5['date'] = df_m5['timestamp_ny'].dt.date
        or_periods = df_m5[(df_m5['timestamp_ny'].dt.hour == 8)]
        or_levels = or_periods.groupby('date').agg({'high': 'max', 'low': 'min'}).to_dict('index')
        df_h1['ema50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
        df_h1_sync = df_h1[['timestamp_ny', 'ema50']].rename(columns={'timestamp_ny': 'h1_time'})
        df_m5 = pd.merge_asof(df_m5.sort_values('timestamp_ny'), df_h1_sync.sort_values('h1_time'), 
                             left_on='timestamp_ny', right_on='h1_time', direction='backward')
        
        i = 20
        total = len(df_m5)
        while i < total:
            row = df_m5.iloc[i]
            curr_time = row.timestamp_ny
            if curr_time.hour < 9 or curr_time.hour >= 13: i += 1; continue
            lvl = or_levels.get(curr_time.date())
            if lvl:
                prev = df_m5.iloc[i-1]
                d = (row.close - row.ema50) * 10000
                if prev.high > lvl['high'] and row.close < lvl['high'] and d > dist_pips:
                    setup = {'direction': 'SHORT', 'entry_p': row.close, 'sl': prev.high + 0.0001, 'tp': row.close - (prev.high - row.close) * tp_r}
                    res = self.resolve(df_m5, i, setup, tp_r)
                    if res is not None: trades.append(res); i += 20; continue
                elif prev.low < lvl['low'] and row.close > lvl['low'] and d < -dist_pips:
                    setup = {'direction': 'LONG', 'entry_p': row.close, 'sl': prev.low - 0.0001, 'tp': row.close + (row.close - prev.low) * tp_r}
                    res = self.resolve(df_m5, i, setup, tp_r)
                    if res is not None: trades.append(res); i += 20; continue
            i += 1
        return pd.Series(trades)

    def resolve(self, df, idx, setup, tp_r):
        sl, tp = setup['sl'], setup['tp']
        for j in range(idx + 1, min(idx + 200, len(df))):
            f = df.iloc[j]
            if setup['direction'] == 'LONG':
                if f.low <= sl: return -1.0
                if f.high >= tp: return tp_r
            else:
                if f.high >= sl: return -1.0
                if f.low <= tp: return tp_r
        return 0.0

def run_deep_matrix():
    print("Phase 10: Deep Matrix - Selective Fakeout")
    engine = Phase10DeepMatrix()
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    p = 'period_2020_2026'
    df_m5 = pd.read_csv(manifest[p]['m5_bid'])
    df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'], utc=True)
    df_h1 = pd.read_csv(manifest[p]['h1_bid'])
    df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True)
    
    matrix = []
    for dist in [15, 20, 25]:
        for tp in [1.5, 2.0, 2.5]:
            print(f"  Testing Dist={dist} TP={tp}...")
            trades = engine.run_variant(df_m5, df_h1, dist, tp)
            if not trades.empty:
                gp = trades[trades > 0].sum()
                gl = abs(trades[trades < 0].sum())
                pf = gp / gl if gl > 0 else 0
                matrix.append({"dist": dist, "tp": tp, "sample": len(trades), "pf": round(pf, 3), "exp": round(trades.mean(), 4)})
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase10_high_frequency_entry_discovery\deep_matrix")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix).to_csv(out_dir / "phase10_deep_matrix_fakeout.csv", index=False)
    print("Deep Matrix Complete.")

if __name__ == "__main__":
    run_deep_matrix()




import pandas as pd
import numpy as np
import json
from pathlib import Path

class FakeoutManagement:
    def __init__(self):
        self.tz_ny = 'America/New_York'

    def run_mgmt(self):
        print("Phase 11: Selective Fakeout Management Optimization")
        manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
        with open(manifest_path, 'r') as f: manifest = json.load(f)
        p = 'period_2020_2026'
        df_m5 = pd.read_csv(manifest[p]['m5_bid'])
        df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'], utc=True)
        df_h1 = pd.read_csv(manifest[p]['h1_bid'])
        df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True)
        
        setups = self.generate_fakeouts(df_m5, df_h1)
        
        matrix = []
        for tp in [2.0, 2.5, 3.0]:
            for be in [None, 1.0, 1.5]:
                print(f"  Testing Fakeout: TP={tp} BE={be}...")
                results = self.simulate(df_m5, setups, tp, be)
                if not results.empty:
                    gp = results[results > 0].sum()
                    gl = abs(results[results < 0].sum())
                    pf = gp / gl if gl > 0 else 0
                    matrix.append({"tp": tp, "be": be, "pf": round(pf, 3), "sample": len(results)})
        
        out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase11_two_entries_management\new_methods_management")
        pd.DataFrame(matrix).to_csv(out_dir / "fakeout_management_matrix.csv", index=False)

    def generate_fakeouts(self, df_m5, df_h1):
        # Phase 10 Selective Fakeout Logic
        df_m5 = df_m5.copy()
        df_m5['timestamp_ny'] = df_m5['timestamp'].dt.tz_convert(self.tz_ny)
        df_h1 = df_h1.copy()
        df_h1['timestamp_ny'] = df_h1['timestamp'].dt.tz_convert(self.tz_ny)
        df_m5['date'] = df_m5['timestamp_ny'].dt.date
        or_periods = df_m5[(df_m5['timestamp_ny'].dt.hour == 8)]
        or_levels = or_periods.groupby('date').agg({'high': 'max', 'low': 'min'}).to_dict('index')
        df_h1['ema50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
        df_h1_sync = df_h1[['timestamp_ny', 'ema50']].rename(columns={'timestamp_ny': 'h1_time'})
        df_m5 = pd.merge_asof(df_m5.sort_values('timestamp_ny'), df_h1_sync.sort_values('h1_time'), 
                             left_on='timestamp_ny', right_on='h1_time', direction='backward')
        
        setups = []
        for i in range(100, len(df_m5)):
            row = df_m5.iloc[i]
            if row.timestamp_ny.hour < 9 or row.timestamp_ny.hour >= 13: continue
            lvl = or_levels.get(row.timestamp_ny.date())
            if lvl:
                prev = df_m5.iloc[i-1]
                d = (row.close - row.ema50) * 10000
                if prev.high > lvl['high'] and row.close < lvl['high'] and d > 20:
                    setups.append({'m5_idx': i, 'direction': 'SHORT', 'entry_p': row.close, 'sl': prev.high + 0.0001})
                elif prev.low < lvl['low'] and row.close > lvl['low'] and d < -20:
                    setups.append({'m5_idx': i, 'direction': 'LONG', 'entry_p': row.close, 'sl': prev.low - 0.0001})
        return pd.DataFrame(setups)

    def simulate(self, df, setups, tp_r, be_r):
        results = []
        for idx, s in setups.iterrows():
            entry_idx = s['m5_idx']
            sl = s['sl']
            tp = s['entry_p'] + (s['entry_p'] - sl) * tp_r if s['direction'] == 'LONG' else s['entry_p'] - (sl - s['entry_p']) * tp_r
            be_trigger = s['entry_p'] + (s['entry_p'] - sl) * be_r if be_r else None
            if s['direction'] == 'SHORT' and be_r: be_trigger = s['entry_p'] - (sl - s['entry_p']) * be_r
            
            be_active = False
            r_val = -1.0
            for j in range(entry_idx + 1, min(entry_idx + 150, len(df))):
                f = df.iloc[j]
                if be_r and not be_active:
                    if s['direction'] == 'LONG' and f.high >= be_trigger: be_active = True
                    elif s['direction'] == 'SHORT' and f.low <= be_trigger: be_active = True
                
                if s['direction'] == 'LONG':
                    if f.low <= (s['entry_p'] if be_active else sl): r_val = 0.0 if be_active else -1.0; break
                    if f.high >= tp: r_val = tp_r; break
                else:
                    if f.high >= (s['entry_p'] if be_active else sl): r_val = 0.0 if be_active else -1.0; break
                    if f.low <= tp: r_val = tp_r; break
            results.append(r_val)
        return pd.Series(results)

if __name__ == "__main__":
    eng = FakeoutManagement()
    eng.run_mgmt()



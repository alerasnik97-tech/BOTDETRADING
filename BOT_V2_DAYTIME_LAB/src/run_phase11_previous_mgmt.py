
import pandas as pd
import numpy as np
import json
from pathlib import Path

class Phase11PreviousManagement:
    def __init__(self):
        self.tz_ny = 'America/New_York'

    def run_retest(self):
        print("Phase 11: Retesting Previous Candidates with New Management")
        # Load Phase 8 Setups
        # (Assuming we have them or can regenerate them)
        # For Phase 11, I will re-run the Phase 8 logic and apply management
        
        manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
        with open(manifest_path, 'r') as f: manifest = json.load(f)
        p = 'period_2020_2026'
        df_m5 = pd.read_csv(manifest[p]['m5_bid'])
        df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'], utc=True)
        
        candidates = [
            {"id": "Phase8", "name": "High_Precision", "n": 8, "body": 0.6},
            {"id": "Phase7", "name": "Balanced", "n": 8, "body": 0.0},
            {"id": "Phase10", "name": "Selective_Fakeout", "type": "fakeout"}
        ]
        
        matrix = []
        for cand in candidates:
            print(f"  Retesting {cand['name']}...")
            setups = self.generate_setups(df_m5, cand)
            
            for tp in [1.5, 2.0, 2.5]:
                for be in [None, 0.75, 1.0]:
                    print(f"    Testing {cand['name']}: TP={tp} BE={be}...")
                    results = self.simulate(df_m5, setups, tp, be)
                    if not results.empty:
                        gp = results[results > 0].sum()
                        gl = abs(results[results < 0].sum())
                        pf = gp / gl if gl > 0 else 0
                        matrix.append({"candidate": cand['name'], "tp": tp, "be": be, "pf": round(pf, 3), "sample": len(results)})
        
        out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase11_two_entries_management\previous_candidates_management")
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(matrix).to_csv(out_dir / "previous_candidates_management_matrix.csv", index=False)

    def generate_setups(self, df, cand):
        # Implementation of candidate setup logic
        setups = []
        df_m5 = df.copy()
        df_m5['timestamp_ny'] = df_m5['timestamp'].dt.tz_convert(self.tz_ny)
        
        if cand['id'] == "Phase8" or cand['id'] == "Phase7":
            n = cand['n']
            df_m5['h_sweep'] = (df_m5['high'] > df_m5['high'].shift(1).rolling(n).max())
            df_m5['l_sweep'] = (df_m5['low'] < df_m5['low'].shift(1).rolling(n).min())
            
            for i in range(n, len(df_m5)):
                row = df_m5.iloc[i]
                if row.timestamp_ny.hour < 8 or row.timestamp_ny.hour >= 12: continue
                if cand['id'] == "Phase8" and row.timestamp_ny.weekday() >= 4: continue # No Friday
                
                body = abs(row.close - row.open)
                rng = row.high - row.low
                if rng == 0: continue
                
                if row.h_sweep and row.close < row.open and body/rng >= cand['body']:
                    setups.append({'m5_idx': i, 'direction': 'SHORT', 'entry_p': row.close, 'sl': row.high + 0.0001})
                elif row.l_sweep and row.close > row.open and body/rng >= cand['body']:
                    setups.append({'m5_idx': i, 'direction': 'LONG', 'entry_p': row.close, 'sl': row.low - 0.0001})
        
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
            for j in range(entry_idx + 1, min(entry_idx + 100, len(df))):
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
    eng = Phase11PreviousManagement()
    eng.run_retest()



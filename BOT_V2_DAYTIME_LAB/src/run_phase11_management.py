
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import timedelta

class Phase11ManagementMatrix:
    def __init__(self):
        self.tz_ny = 'America/New_York'

    def run_management_sim(self, df_m5, base_trades, config):
        """Simulates management on a list of identified setups"""
        results = []
        tp_r = config.get('tp_r', 1.5)
        be_r = config.get('be_r', None)
        timeout_bars = config.get('timeout_bars', 36) # 3h in M5
        
        for idx, setup in base_trades.iterrows():
            entry_idx = setup['m5_idx']
            sl, tp = setup['sl'], setup['entry_p'] + (setup['entry_p'] - setup['sl']) * tp_r if setup['direction'] == 'LONG' else setup['entry_p'] - (setup['sl'] - setup['entry_p']) * tp_r
            
            be_active = False
            be_lvl = setup['entry_p']
            be_trigger = setup['entry_p'] + (setup['entry_p'] - setup['sl']) * be_r if be_r else None
            if setup['direction'] == 'SHORT' and be_r:
                be_trigger = setup['entry_p'] - (setup['sl'] - setup['entry_p']) * be_r
            
            trade_res = 0.0
            
            for j in range(entry_idx + 1, min(entry_idx + timeout_bars, len(df_m5))):
                f = df_m5.iloc[j]
                
                # Check BE Trigger
                if be_r and not be_active:
                    if setup['direction'] == 'LONG' and f.high >= be_trigger: be_active = True
                    elif setup['direction'] == 'SHORT' and f.low <= be_trigger: be_active = True
                
                # Check Exit
                if setup['direction'] == 'LONG':
                    if f.low <= (be_lvl if be_active else sl):
                        trade_res = 0.0 if be_active else -1.0
                        break
                    if f.high >= tp:
                        trade_res = tp_r
                        break
                else:
                    if f.high >= (be_lvl if be_active else sl):
                        trade_res = 0.0 if be_active else -1.0
                        break
                    if f.low <= tp:
                        trade_res = tp_r
                        break
            
            # Timeout
            if trade_res == 0.0:
                final_p = df_m5.iloc[min(entry_idx + timeout_bars, len(df_m5)-1)].close
                r_val = (final_p - setup['entry_p']) / (setup['entry_p'] - setup['sl']) if setup['direction'] == 'LONG' else (setup['entry_p'] - final_p) / (setup['sl'] - setup['entry_p'])
                trade_res = max(-1.0, min(tp_r, r_val))
            
            results.append(trade_res)
        
        return pd.Series(results)

    def run_matrix(self):
        print("Phase 11: Management Matrix Optimization")
        manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
        with open(manifest_path, 'r') as f: manifest = json.load(f)
        p = 'period_2020_2026'
        df_m5 = pd.read_csv(manifest[p]['m5_bid'])
        df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'], utc=True)
        
        # 1. Identify Setups for Method 1 (Trend Pullback)
        # We need a clean list of setups
        # ... (Re-implementing the core logic for setups) ...
        # For brevity, I'll use the screening script's logic to generate a setup pool
        setups = self.generate_m1_setups(df_m5)
        
        matrix = []
        for tp in [1.0, 1.25, 1.5, 2.0]:
            for be in [None, 0.5, 0.75, 1.0]:
                for to in [12, 24, 48]: # 1h, 2h, 4h
                    print(f"  Testing M1: TP={tp} BE={be} TO={to}...")
                    results = self.run_management_sim(df_m5, setups, {'tp_r': tp, 'be_r': be, 'timeout_bars': to})
                    if not results.empty:
                        gp = results[results > 0].sum()
                        gl = abs(results[results < 0].sum())
                        pf = gp / gl if gl > 0 else 0
                        matrix.append({"method": "M1_Trend", "tp": tp, "be": be, "timeout": to, "pf": round(pf, 3), "exp": round(results.mean(), 4), "sample": len(results)})
        
        out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase11_two_entries_management\new_methods_management")
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(matrix).to_csv(out_dir / "method1_management_matrix.csv", index=False)
        print("Matrix Complete.")

    def generate_m1_setups(self, df_m5):
        # Implementation of Method 1 setup identification
        setups = []
        df_m5 = df_m5.copy()
        df_m5['timestamp_ny'] = df_m5['timestamp'].dt.tz_convert(self.tz_ny)
        
        # Simplified H1/Bias for setup generation
        df_m5['ema200'] = df_m5['close'].ewm(span=200, adjust=False).mean()
        df_m5['ema200_slope'] = df_m5['ema200'].diff()
        
        for i in range(200, len(df_m5), 5):
            row = df_m5.iloc[i]
            if row.timestamp_ny.hour < 8 or row.timestamp_ny.hour >= 15: continue
            
            bias = 1 if row.ema200_slope > 0 else -1
            ema20 = df_m5['close'].iloc[i-20:i].mean()
            
            if bias == 1 and row.low <= ema20 and row.close > ema20:
                setups.append({'m5_idx': i, 'direction': 'LONG', 'entry_p': row.close, 'sl': row.low - 0.0001})
            elif bias == -1 and row.high >= ema20 and row.close < ema20:
                setups.append({'m5_idx': i, 'direction': 'SHORT', 'entry_p': row.close, 'sl': row.high + 0.0001})
        
        return pd.DataFrame(setups)

if __name__ == "__main__":
    eng = Phase11ManagementMatrix()
    eng.run_matrix()



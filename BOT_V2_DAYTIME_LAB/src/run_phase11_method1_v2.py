
import pandas as pd
import numpy as np
import json
from pathlib import Path

class Method1V2Engine:
    def __init__(self):
        self.tz_ny = 'America/New_York'

    def run_screening(self, df_m5):
        print("Method 1 V2: H1 Trend + M5 FVG")
        trades = []
        df_m5 = df_m5.copy()
        df_m5['timestamp_ny'] = df_m5['timestamp'].dt.tz_convert(self.tz_ny)
        
        # Bias H1: Simplified
        df_m5['ema200'] = df_m5['close'].ewm(span=200, adjust=False).mean()
        df_m5['ema200_slope'] = df_m5['ema200'].diff()
        
        for i in range(10, len(df_m5) - 1):
            row = df_m5.iloc[i]
            if row.timestamp_ny.hour < 8 or row.timestamp_ny.hour >= 15: continue
            
            bias = 1 if row.ema200_slope > 0 else -1
            
            # FVG Detection (M5)
            # Bullish FVG: Low(i) > High(i-2)
            # Bearish FVG: High(i) < Low(i-2)
            
            p2 = df_m5.iloc[i-2]
            p1 = df_m5.iloc[i-1]
            
            if bias == 1:
                if row.low > p2.high: # Bullish FVG
                    # Entry on retest of FVG or next candle
                    setup = {'direction': 'LONG', 'entry_p': row.close, 'sl': p2.high - 0.0001, 'tp': row.close + (row.close - p2.high) * 2.0}
                    res = self.resolve(df_m5, i, setup)
                    if res is not None: trades.append(res); i += 20; continue
            elif bias == -1:
                if row.high < p2.low: # Bearish FVG
                    setup = {'direction': 'SHORT', 'entry_p': row.close, 'sl': p2.low + 0.0001, 'tp': row.close - (p2.low - row.close) * 2.0}
                    res = self.resolve(df_m5, i, setup)
                    if res is not None: trades.append(res); i += 20; continue
            i += 1
        return pd.Series(trades)

    def resolve(self, df, idx, setup):
        sl, tp = setup['sl'], setup['tp']
        for j in range(idx + 1, min(idx + 150, len(df))):
            f = df.iloc[j]
            if setup['direction'] == 'LONG':
                if f.low <= sl: return -1.0
                if f.high >= tp: return 2.0
            else:
                if f.high >= sl: return -1.0
                if f.low <= tp: return 2.0
        return 0.0

if __name__ == "__main__":
    eng = Method1V2Engine()
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    p = 'period_2020_2026'
    df_m5 = pd.read_csv(manifest[p]['m5_bid'])
    df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'], utc=True)
    res = eng.run_screening(df_m5)
    if not res.empty:
        pf = res[res > 0].sum() / abs(res[res < 0].sum())
        print(f"Result: Sample={len(res)} PF={pf:.3f} Exp={res.mean():.4f}")



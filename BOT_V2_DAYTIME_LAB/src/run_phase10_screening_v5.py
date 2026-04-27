
import pandas as pd
import numpy as np
import json
from pathlib import Path

class Phase10DiscoveryEngineV5:
    def __init__(self):
        self.tz_ny = 'America/New_York'

    def run_family_screening(self, df_m5, df_h1, news_df):
        trades = []
        df_m5 = df_m5.copy()
        df_m5['timestamp_ny'] = df_m5['timestamp'].dt.tz_convert(self.tz_ny)
        df_h1['timestamp_ny'] = df_h1['timestamp'].dt.tz_convert(self.tz_ny)
        
        # Pre-calculate OR (08:00 - 09:00)
        df_m5['date'] = df_m5['timestamp_ny'].dt.date
        or_periods = df_m5[(df_m5['timestamp_ny'].dt.hour == 8)]
        or_levels = or_periods.groupby('date').agg({'high': 'max', 'low': 'min'}).to_dict('index')
        
        # H1 EMA 50 for Filter
        df_h1['ema50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
        df_h1_sync = df_h1[['timestamp_ny', 'ema50']].rename(columns={'timestamp_ny': 'h1_time'})
        df_m5 = pd.merge_asof(df_m5.sort_values('timestamp_ny'), df_h1_sync.sort_values('h1_time'), 
                             left_on='timestamp_ny', right_on='h1_time', direction='backward')
        
        i = 20
        total = len(df_m5)
        while i < total:
            row = df_m5.iloc[i]
            curr_time = row.timestamp_ny
            if curr_time.hour < 9 or curr_time.hour >= 12: i += 1; continue
            
            lvl = or_levels.get(curr_time.date())
            if lvl:
                prev = df_m5.iloc[i-1]
                # FAKEOUT + H1 EMA FILTER
                # If price is far from EMA 50 H1, expect reversion
                dist = (row.close - row.ema50) * 10000
                
                if prev.high > lvl['high'] and row.close < lvl['high'] and dist > 15: # Failed up + overextended
                    setup = {'direction': 'SHORT', 'entry_p': row.close, 'sl': prev.high + 0.0001, 'tp': row.close - (prev.high - row.close) * 2.0}
                    res, _ = self.resolve_trade(df_m5, i, setup)
                    if res is not None: trades.append({'time': curr_time, 'result': res}); i += 20; continue
                elif prev.low < lvl['low'] and row.close > lvl['low'] and dist < -15: # Failed down + overextended
                    setup = {'direction': 'LONG', 'entry_p': row.close, 'sl': prev.low - 0.0001, 'tp': row.close + (row.close - prev.low) * 2.0}
                    res, _ = self.resolve_trade(df_m5, i, setup)
                    if res is not None: trades.append({'time': curr_time, 'result': res}); i += 20; continue
            i += 1
        return pd.DataFrame(trades)

    def resolve_trade(self, df, start_idx, setup):
        sl, tp = setup['sl'], setup['tp']
        for j in range(start_idx + 1, min(start_idx + 200, len(df))):
            f = df.iloc[j]
            if setup['direction'] == 'LONG':
                if f.low <= sl: return -1.0, f.timestamp_ny
                if f.high >= tp: return 2.0, f.timestamp_ny
            else:
                if f.high >= sl: return -1.0, f.timestamp_ny
                if f.low <= tp: return 2.0, f.timestamp_ny
        return 0.0, df.iloc[min(start_idx + 200, len(df)-1)].timestamp_ny

def run_screening():
    print("Phase 10: Discovery Screening V5 (Selective Fakeout)")
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase10_high_frequency_entry_discovery\screening")
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = Phase10DiscoveryEngineV5()
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    periods = ['period_2020_2026']
    for p in periods:
        df_m5 = pd.read_csv(manifest[p]['m5_bid'])
        df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'], utc=True)
        df_h1 = pd.read_csv(manifest[p]['h1_bid'])
        df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True)
        trades = engine.run_family_screening(df_m5, df_h1, None)
        if not trades.empty:
            pf = trades[trades['result'] > 0]['result'].sum() / abs(trades[trades['result'] < 0]['result'].sum()) if any(trades['result'] < 0) else 1.0
            print(f"  Result: Sample={len(trades)} PF={pf:.3f} Exp={trades['result'].mean():.4f}")

if __name__ == "__main__":
    run_screening()



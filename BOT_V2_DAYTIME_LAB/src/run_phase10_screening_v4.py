
import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings

# Suppress reindexing warnings
warnings.filterwarnings('ignore', category=UserWarning)

class Phase10DiscoveryEngineV4:
    def __init__(self):
        self.tz_ny = 'America/New_York'

    def run_family_screening(self, df_m3, df_h1, news_df, family_id, config):
        trades = []
        df_m3 = df_m3.copy()
        df_m3['timestamp_ny'] = df_m3['timestamp'].dt.tz_convert(self.tz_ny)
        df_h1['timestamp_ny'] = df_h1['timestamp'].dt.tz_convert(self.tz_ny)
        
        # Fractals
        n = 3
        df_m3['is_h_frac'] = (df_m3['high'] == df_m3['high'].rolling(window=2*n+1, center=True).max())
        df_m3['is_l_frac'] = (df_m3['low'] == df_m3['low'].rolling(window=2*n+1, center=True).min())
        
        # H1 Levels
        df_h1['date'] = df_h1['timestamp_ny'].dt.date
        levels = df_h1.groupby('date').agg({'high': 'max', 'low': 'min'}).shift(1).to_dict('index')
        
        # Sync H1 Bias
        df_h1['ema50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
        df_h1['ema_slope'] = df_h1['ema50'].diff()
        df_h1_sync = df_h1[['timestamp_ny', 'ema_slope']].rename(columns={'timestamp_ny': 'h1_time'})
        df_m3 = pd.merge_asof(df_m3.sort_values('timestamp_ny'), df_h1_sync.sort_values('h1_time'), 
                             left_on='timestamp_ny', right_on='h1_time', direction='backward')
        
        # News
        t_col = 'timestamp_utc' if 'timestamp_utc' in news_df.columns else 'timestamp'
        news_times = pd.to_datetime(news_df[t_col], utc=True).tolist()
        
        active_sweep = None
        last_trade_date = None
        
        i = 20
        total = len(df_m3)
        while i < total:
            row = df_m3.iloc[i]
            curr_time = row.timestamp_ny
            curr_date = curr_time.date()
            
            if curr_time.hour < 7 or curr_time.hour >= 20: i += 1; continue
            
            lvl = levels.get(curr_date)
            if lvl:
                # 1. Sweep Detection
                if row.high > lvl['high']: active_sweep = {'type': 'H', 'start_idx': i}
                elif row.low < lvl['low']: active_sweep = {'type': 'L', 'start_idx': i}
                
                # 2. CHoCH Detection
                if active_sweep:
                    if (i - active_sweep['start_idx']) > 20:
                        active_sweep = None
                    else:
                        window = df_m3.iloc[max(0, active_sweep['start_idx']-10):i]
                        if active_sweep['type'] == 'H':
                            l_fracs = window[window['is_l_frac']]
                            if not l_fracs.empty:
                                if row.close < l_fracs['low'].iloc[-1]: # CHoCH
                                    sl = df_m3.iloc[active_sweep['start_idx']:i+1]['high'].max() + 0.0001
                                    tp = row.close - (sl - row.close) * 1.5
                                    setup = {'direction': 'SHORT', 'entry_p': row.close, 'sl': sl, 'tp': tp}
                                    res, _ = self.resolve_trade(df_m3, i, setup)
                                    if res is not None: trades.append({'time': curr_time, 'result': res})
                                    active_sweep = None; i += 10; continue
                        else:
                            h_fracs = window[window['is_h_frac']]
                            if not h_fracs.empty:
                                if row.close > h_fracs['high'].iloc[-1]: # CHoCH
                                    sl = df_m3.iloc[active_sweep['start_idx']:i+1]['low'].min() - 0.0001
                                    tp = row.close + (row.close - sl) * 1.5
                                    setup = {'direction': 'LONG', 'entry_p': row.close, 'sl': sl, 'tp': tp}
                                    res, _ = self.resolve_trade(df_m3, i, setup)
                                    if res is not None: trades.append({'time': curr_time, 'result': res})
                                    active_sweep = None; i += 10; continue
            i += 1
        return pd.DataFrame(trades)

    def resolve_trade(self, df, start_idx, setup):
        sl, tp = setup['sl'], setup['tp']
        for j in range(start_idx + 1, min(start_idx + 150, len(df))):
            f = df.iloc[j]
            if setup['direction'] == 'LONG':
                if f.low <= sl: return -1.0, f.timestamp_ny
                if f.high >= tp: return 1.5, f.timestamp_ny
            else:
                if f.high >= sl: return -1.0, f.timestamp_ny
                if f.low <= tp: return 1.5, f.timestamp_ny
        return 0.0, df.iloc[min(start_idx + 150, len(df)-1)].timestamp_ny

def run_screening():
    print("Phase 10: Discovery Screening V4.1 (Clean)")
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase10_high_frequency_entry_discovery\screening")
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = Phase10DiscoveryEngineV4()
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    periods = ['period_2015_2019', 'period_2020_2026']
    
    results = []
    for p in periods:
        print(f"  Processing Period: {p}...")
        df_src = pd.read_csv(manifest[p]['m5_bid'])
        df_src['timestamp'] = pd.to_datetime(df_src['timestamp'], utc=True)
        df_src.set_index('timestamp', inplace=True)
        df_m3 = df_src.resample('3min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna().reset_index()
        df_h1 = pd.read_csv(manifest[p]['h1_bid'])
        df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True)
        news_df = pd.read_csv(manifest[p]['news'])
        
        trades = engine.run_family_screening(df_m3, df_h1, news_df, 2, {})
        if not trades.empty:
            gp = trades[trades['result'] > 0]['result'].sum()
            gl = abs(trades[trades['result'] < 0]['result'].sum())
            pf = gp / gl if gl > 0 else 0
            res = {"period": p, "family": "Fast_CHoCH_N3_M3", "sample": len(trades), "pf": round(pf, 3), "expectancy": round(trades['result'].mean(), 4)}
            results.append(res)
            print(f"    Result: {res}")
            
    pd.DataFrame(results).to_csv(out_dir / "phase10_family_screening_v4_results.csv", index=False)

if __name__ == "__main__":
    run_screening()



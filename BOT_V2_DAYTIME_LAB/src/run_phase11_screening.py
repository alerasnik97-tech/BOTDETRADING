
import pandas as pd
import numpy as np
import json
from pathlib import Path

class Phase11DiscoveryEngine:
    def __init__(self):
        self.tz_ny = 'America/New_York'

    def run_method1_pullback(self, df_m5, df_h1, news_df, config):
        """H1 Directional Pullback + LTF Rejection"""
        trades = []
        df_m5 = df_m5.copy()
        df_m5['timestamp_ny'] = df_m5['timestamp'].dt.tz_convert(self.tz_ny)
        df_h1['timestamp_ny'] = df_h1['timestamp'].dt.tz_convert(self.tz_ny)
        
        # H1 Bias: EMA 50 Slope
        df_h1['ema50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
        df_h1['ema_slope'] = df_h1['ema50'].diff()
        
        df_h1_sync = df_h1[['timestamp_ny', 'ema50', 'ema_slope']].rename(columns={'timestamp_ny': 'h1_time'})
        df_m5 = pd.merge_asof(df_m5.sort_values('timestamp_ny'), df_h1_sync.sort_values('h1_time'), 
                             left_on='timestamp_ny', right_on='h1_time', direction='backward')
        
        t_col = 'timestamp_utc' if 'timestamp_utc' in news_df.columns else 'timestamp'
        news_times = pd.to_datetime(news_df[t_col], utc=True).tolist()
        
        last_trade_date = None
        i = 20
        total = len(df_m5)
        while i < total:
            row = df_m5.iloc[i]
            curr_time = row.timestamp_ny
            if curr_time.hour < 7 or curr_time.hour >= 16: i += 1; continue
            
            # Bias H1
            bias = 1 if row.ema_slope > 0 and row.close > row.ema50 else (-1 if row.ema_slope < 0 and row.close < row.ema50 else 0)
            
            if bias != 0:
                # M5 Pullback to EMA 20
                ema20_m5 = df_m5['close'].iloc[i-20:i].mean()
                body = abs(row.close - row.open)
                rng = row.high - row.low
                
                # Rejection Rule
                if bias == 1 and row.low <= ema20_m5 and row.close > ema20_m5 and body/rng > 0.5:
                    setup = {'direction': 'LONG', 'entry_p': row.close, 'sl': row.low - 0.0001, 'tp': row.close + (row.close - row.low) * 1.5}
                    res = self.resolve(df_m5, i, setup)
                    if res is not None: trades.append({'time': curr_time, 'result': res}); i += 20; continue
                elif bias == -1 and row.high >= ema20_m5 and row.close < ema20_m5 and body/rng > 0.5:
                    setup = {'direction': 'SHORT', 'entry_p': row.close, 'sl': row.high + 0.0001, 'tp': row.close - (row.high - row.close) * 1.5}
                    res = self.resolve(df_m5, i, setup)
                    if res is not None: trades.append({'time': curr_time, 'result': res}); i += 20; continue
            i += 1
        return pd.DataFrame(trades)

    def run_method2_axis(self, df_m5, df_h1, news_df, config):
        """H1 Extension + LTF Mean Reversion to Axis"""
        trades = []
        df_m5 = df_m5.copy()
        df_m5['timestamp_ny'] = df_m5['timestamp'].dt.tz_convert(self.tz_ny)
        df_h1['timestamp_ny'] = df_h1['timestamp'].dt.tz_convert(self.tz_ny)
        df_h1['ema50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
        df_h1_sync = df_h1[['timestamp_ny', 'ema50']].rename(columns={'timestamp_ny': 'h1_time'})
        df_m5 = pd.merge_asof(df_m5.sort_values('timestamp_ny'), df_h1_sync.sort_values('h1_time'), 
                             left_on='timestamp_ny', right_on='h1_time', direction='backward')
        
        i = 20
        total = len(df_m5)
        while i < total:
            row = df_m5.iloc[i]
            curr_time = row.timestamp_ny
            if curr_time.hour < 9 or curr_time.hour >= 14: i += 1; continue
            
            pips_dist = (row.close - row.ema50) * 10000
            if abs(pips_dist) > 25: # Overextended
                if pips_dist > 25 and row.close < row.open: # Bearish rejection
                    setup = {'direction': 'SHORT', 'entry_p': row.close, 'sl': row.high + 0.0001, 'tp': row.ema50}
                    res = self.resolve(df_m5, i, setup)
                    if res is not None: trades.append({'time': curr_time, 'result': res}); i += 20; continue
                elif pips_dist < -25 and row.close > row.open: # Bullish rejection
                    setup = {'direction': 'LONG', 'entry_p': row.close, 'sl': row.low - 0.0001, 'tp': row.ema50}
                    res = self.resolve(df_m5, i, setup)
                    if res is not None: trades.append({'time': curr_time, 'result': res}); i += 20; continue
            i += 1
        return pd.DataFrame(trades)

    def resolve(self, df, idx, setup):
        sl, tp = setup['sl'], setup['tp']
        for j in range(idx + 1, min(idx + 150, len(df))):
            f = df.iloc[j]
            if setup['direction'] == 'LONG':
                if f.low <= sl: return -1.0
                if f.high >= tp: return 1.5
            else:
                if f.high >= sl: return -1.0
                if f.low <= tp: return 1.5
        return 0.0

def run_screening():
    print("Phase 11: Screening New Methods")
    engine = Phase11DiscoveryEngine()
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    p = 'period_2020_2026'
    df_m5 = pd.read_csv(manifest[p]['m5_bid'])
    df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'], utc=True)
    df_h1 = pd.read_csv(manifest[p]['h1_bid'])
    df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True)
    news_df = pd.read_csv(manifest[p]['news'])
    
    results = []
    
    print("  Method 1: Trend Pullback...")
    trades1 = engine.run_method1_pullback(df_m5, df_h1, news_df, {})
    if not trades1.empty:
        pf = trades1[trades1['result'] > 0]['result'].sum() / abs(trades1[trades1['result'] < 0]['result'].sum())
        results.append({"method": "Trend_Pullback", "sample": len(trades1), "pf": round(pf, 3), "exp": round(trades1['result'].mean(), 4)})
        
    print("  Method 2: Axis Reversion...")
    trades2 = engine.run_method2_axis(df_m5, df_h1, news_df, {})
    if not trades2.empty:
        pf = (trades2[trades2['result'] > 0]['result'].count() * 1.0) / trades2[trades2['result'] < 0]['result'].count() if any(trades2['result'] < 0) else 1.0
        results.append({"method": "Axis_Reversion", "sample": len(trades2), "pf": round(pf, 3), "exp": round(trades2['result'].mean(), 4)})

    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase11_two_entries_management\screening")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "phase11_new_methods_screening.csv", index=False)
    print("Screening Complete.")

if __name__ == "__main__":
    run_screening()



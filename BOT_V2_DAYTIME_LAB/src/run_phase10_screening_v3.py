
import pandas as pd
import numpy as np
import json
from pathlib import Path

class Phase10DiscoveryEngineV3:
    def __init__(self):
        self.tz_ny = 'America/New_York'

    def run_family_screening(self, df_m5, df_h1, news_df, family_id, config):
        trades = []
        df_m5 = df_m5.copy()
        df_m5['timestamp_ny'] = df_m5['timestamp'].dt.tz_convert(self.tz_ny)
        df_h1['timestamp_ny'] = df_h1['timestamp'].dt.tz_convert(self.tz_ny)
        
        # Context
        df_h1['ema50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
        df_h1['ema_slope'] = df_h1['ema50'].diff()
        
        # News
        t_col = 'timestamp_utc' if 'timestamp_utc' in news_df.columns else 'timestamp'
        news_df[t_col] = pd.to_datetime(news_df[t_col], utc=True)
        news_times = news_df[t_col].tolist()
        
        # Pre-calculate OR (08:00 - 09:00)
        df_m5['date'] = df_m5['timestamp_ny'].dt.date
        or_periods = df_m5[(df_m5['timestamp_ny'].dt.hour == 8)]
        or_levels = or_periods.groupby('date').agg({'high': 'max', 'low': 'min'}).to_dict('index')
        
        # Sync H1
        df_h1_sync = df_h1[['timestamp_ny', 'ema50', 'ema_slope']].rename(columns={'timestamp_ny': 'h1_time'})
        df_m5 = pd.merge_asof(df_m5.sort_values('timestamp_ny'), df_h1_sync.sort_values('h1_time'), 
                             left_on='timestamp_ny', right_on='h1_time', direction='backward')
            
        last_trade_date = None
        
        i = 20
        total = len(df_m5)
        while i < total:
            row = df_m5.iloc[i]
            curr_time = row.timestamp_ny
            curr_date = curr_time.date()
            
            if curr_time.hour < 7 or curr_time.hour >= 20: i += 1; continue
            if config.get('one_trade_per_day') and last_trade_date == curr_date: i += 1; continue
            
            setup = None
            lvl = or_levels.get(curr_date)
            
            # FAMILY 4: ORB Bias
            if family_id == 4:
                if curr_time.hour == 9 and curr_time.minute == 35 and lvl:
                    bias = 1 if row.ema_slope > 0 else -1
                    if row.close > lvl['high'] and bias == 1:
                        setup = {'direction': 'LONG', 'entry_p': row.close, 'sl': lvl['low'], 'tp': row.close + (row.close - lvl['low']) * 1.5}
                    elif row.close < lvl['low'] and bias == -1:
                        setup = {'direction': 'SHORT', 'entry_p': row.close, 'sl': lvl['high'], 'tp': row.close - (lvl['high'] - row.close) * 1.5}

            # FAMILY 5: Failed ORB
            elif family_id == 5:
                if curr_time.hour >= 9 and curr_time.hour < 12 and lvl:
                    prev_row = df_m5.iloc[i-1]
                    if prev_row.high > lvl['high'] and row.close < lvl['high']:
                        setup = {'direction': 'SHORT', 'entry_p': row.close, 'sl': prev_row.high + 0.0001, 'tp': row.close - (prev_row.high - row.close) * 1.5}
                    elif prev_row.low < lvl['low'] and row.close > lvl['low']:
                        setup = {'direction': 'LONG', 'entry_p': row.close, 'sl': prev_row.low - 0.0001, 'tp': row.close + (row.close - prev_row.low) * 1.5}

            # FAMILY 6: Axis Reversion
            elif family_id == 6:
                pips = (row.close - row.ema50) * 10000
                if abs(pips) > 25:
                    if pips > 25 and row.close < row.open:
                        setup = {'direction': 'SHORT', 'entry_p': row.close, 'sl': row.high + 0.0001, 'tp': row.ema50}
                    elif pips < -25 and row.close > row.open:
                        setup = {'direction': 'LONG', 'entry_p': row.close, 'sl': row.low - 0.0001, 'tp': row.ema50}

            if setup:
                in_news = any(abs((curr_time - nt).total_seconds()) < 30*60 for nt in news_times)
                if not in_news:
                    res, exit_t = self.resolve_trade(df_m5, i, setup, config)
                    if res is not None:
                        trades.append({'time': curr_time, 'result': res})
                        last_trade_date = curr_date
                        i += 10; continue
            i += 1
        return pd.DataFrame(trades)

    def resolve_trade(self, df, start_idx, setup, config):
        sl, tp = setup['sl'], setup['tp']
        for j in range(start_idx + 1, min(start_idx + 150, len(df))):
            f = df.iloc[j]
            if setup['direction'] == 'LONG':
                if f.low <= sl: return -1.0, f.timestamp_ny
                if f.high >= tp: return 1.5 if setup['tp'] != setup.get('ema50') else 1.0, f.timestamp_ny
            else:
                if f.high >= sl: return -1.0, f.timestamp_ny
                if f.low <= tp: return 1.5 if setup['tp'] != setup.get('ema50') else 1.0, f.timestamp_ny
        return 0.0, df.iloc[min(start_idx + 150, len(df)-1)].timestamp_ny

def run_screening():
    print("Phase 10: Discovery Screening V3.1 (Optimized)")
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase10_high_frequency_entry_discovery\screening")
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = Phase10DiscoveryEngineV3()
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    periods = ['period_2015_2019', 'period_2020_2026']
    
    results = []
    for p in periods:
        print(f"  Processing Period: {p}...")
        df_m5 = pd.read_csv(manifest[p]['m5_bid'])
        df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'], utc=True)
        df_h1 = pd.read_csv(manifest[p]['h1_bid'])
        df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True)
        news_df = pd.read_csv(manifest[p]['news'])
        
        families = [{"id": 4, "name": "ORB_Bias"}, {"id": 5, "name": "ORB_Fakeout"}, {"id": 6, "name": "Axis_Reversion"}]
        for fam in families:
            print(f"    Screening Family: {fam['name']}...")
            trades = engine.run_family_screening(df_m5, df_h1, news_df, fam['id'], {})
            if not trades.empty:
                gp = trades[trades['result'] > 0]['result'].sum()
                gl = abs(trades[trades['result'] < 0]['result'].sum())
                pf = gp / gl if gl > 0 else 0
                res = {"period": p, "family": fam['name'], "sample": len(trades), "pf": round(pf, 3), "expectancy": round(trades['result'].mean(), 4)}
                results.append(res)
                print(f"      Result: {res}")
            
    pd.DataFrame(results).to_csv(out_dir / "phase10_family_screening_v3_results.csv", index=False)

if __name__ == "__main__":
    run_screening()



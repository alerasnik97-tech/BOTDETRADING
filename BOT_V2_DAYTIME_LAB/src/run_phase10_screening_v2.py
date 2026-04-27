
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import timedelta

class Phase10DiscoveryEngine:
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
        news_df['timestamp_utc'] = pd.to_datetime(news_df['timestamp_utc'], utc=True)
        news_times = news_df['timestamp_utc'].tolist()
        
        # Levels
        df_h1['date'] = df_h1['timestamp_ny'].dt.date
        levels = df_h1.groupby('date').agg({'high': 'max', 'low': 'min'}).shift(1).to_dict('index')
        
        # Merge H1 context
        df_h1_sync = df_h1[['timestamp_ny', 'ema50', 'ema_slope']].rename(columns={'timestamp_ny': 'h1_time'})
        df_m5 = pd.merge_asof(df_m5.sort_values('timestamp_ny'), df_h1_sync.sort_values('h1_time'), 
                             left_on='timestamp_ny', right_on='h1_time', direction='backward')
            
        last_trade_date = None
        
        i = 0
        total = len(df_m5)
        while i < total:
            if i % 50000 == 0: print(f"    Progress: {i}/{total} bars...")
            row = df_m5.iloc[i]
            curr_time = row.timestamp_ny
            curr_date = curr_time.date()
            
            if curr_time.hour < 7 or curr_time.hour >= 20: 
                i += 1; continue
            if config.get('one_trade_per_day') and last_trade_date == curr_date:
                i += 1; continue
                
            setup = None
            
            # FAMILY 1: Trend Pullback
            if family_id == 1:
                bias = 1 if row.ema_slope > 0 and row.close > row.ema50 else (-1 if row.ema_slope < 0 and row.close < row.ema50 else 0)
                if bias != 0:
                    ema20 = df_m5['close'].iloc[max(0, i-20):i].mean()
                    if bias == 1 and row.low <= ema20 and row.close > ema20:
                        setup = {'direction': 'LONG', 'entry_p': row.close, 'sl': row.low - 0.0001, 'tp': row.close + (row.close - row.low + 0.0001) * 1.5}
                    elif bias == -1 and row.high >= ema20 and row.close < ema20:
                        setup = {'direction': 'SHORT', 'entry_p': row.close, 'sl': row.high + 0.0001, 'tp': row.close - (row.high + 0.0001 - row.close) * 1.5}

            # FAMILY 3: Displacement Candle (M5)
            elif family_id == 3:
                lvl = levels.get(curr_date)
                if lvl:
                    if row.high > lvl['high'] or row.low < lvl['low']:
                        body = abs(row.close - row.open)
                        rng = row.high - row.low
                        if rng > 0 and body / rng >= 0.6:
                            if row.high > lvl['high'] and row.close < row.open:
                                setup = {'direction': 'SHORT', 'entry_p': row.close, 'sl': row.high + 0.0001, 'tp': row.close - (row.high - row.close + 0.0001) * 1.5}
                            elif row.low < lvl['low'] and row.close > row.open:
                                setup = {'direction': 'LONG', 'entry_p': row.close, 'sl': row.low - 0.0001, 'tp': row.close + (row.close - row.low + 0.0001) * 1.5}

            # FAMILY 4: ORB
            elif family_id == 4:
                if curr_time.hour == 9 and curr_time.minute == 35:
                    or_data = df_m5[(df_m5['timestamp_ny'].dt.date == curr_date) & (df_m5['timestamp_ny'].dt.hour >= 8) & (df_m5['timestamp_ny'].dt.hour < 9)]
                    if not or_data.empty:
                        or_h, or_l = or_data['high'].max(), or_data['low'].min()
                        if row.close > or_h:
                            setup = {'direction': 'LONG', 'entry_p': row.close, 'sl': or_l, 'tp': row.close + (row.close - or_l) * 1.0}
                        elif row.close < or_l:
                            setup = {'direction': 'SHORT', 'entry_p': row.close, 'sl': or_h, 'tp': row.close - (or_h - row.close) * 1.0}

            if setup:
                in_news = any(abs((curr_time - nt).total_seconds()) < 30*60 for nt in news_times)
                if not in_news:
                    res, exit_t = self.resolve_trade(df_m5, i, setup, config)
                    if res is not None:
                        trades.append({'time': curr_time, 'family': family_id, 'result': res})
                        last_trade_date = curr_date
                        # We don't skip i here to allow other families to trigger if in the same loop, but each family runs separately anyway.

            i += 1
        return pd.DataFrame(trades)

    def resolve_trade(self, df, start_idx, setup, config):
        tp_r = config.get('tp_r', 1.5)
        sl = setup['sl']
        tp = setup['tp']
        for j in range(start_idx + 1, min(start_idx + 100, len(df))): # Limit to 100 bars for screening
            f = df.iloc[j]
            if setup['direction'] == 'LONG':
                if f.low <= sl: return -1.0, f.timestamp_ny
                if f.high >= tp: return tp_r, f.timestamp_ny
            else:
                if f.high >= sl: return -1.0, f.timestamp_ny
                if f.low <= tp: return tp_r, f.timestamp_ny
        return 0.0, df.iloc[min(start_idx + 100, len(df)-1)].timestamp_ny

def run_screening():
    print("Phase 10: Discovery Screening - V2")
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase10_high_frequency_entry_discovery\screening")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    engine = Phase10DiscoveryEngine()
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    p = 'period_2020_2026'
    df_m5 = pd.read_csv(manifest[p]['m5_bid']).head(20000)
    df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'], utc=True)
    df_h1 = pd.read_csv(manifest[p]['h1_bid'])
    df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True)
    news_df = pd.read_csv(manifest[p]['news'])
    
    families = [
        {"id": 1, "name": "Trend_Pullback", "config": {"tp_r": 1.5, "one_trade_per_day": False}},
        {"id": 3, "name": "Displacement_Sweep", "config": {"tp_r": 1.5, "one_trade_per_day": False}},
        {"id": 4, "name": "ORB_0800_0900", "config": {"tp_r": 1.0, "one_trade_per_day": True}}
    ]
    
    results = []
    for fam in families:
        print(f"  Screening Family: {fam['name']}...")
        trades = engine.run_family_screening(df_m5, df_h1, news_df, fam['id'], fam['config'])
        if not trades.empty:
            gp = trades[trades['result'] > 0]['result'].sum()
            gl = abs(trades[trades['result'] < 0]['result'].sum())
            pf = gp / gl if gl > 0 else 0
            res = {
                "family": fam['name'],
                "sample": len(trades),
                "pf": round(pf, 3),
                "expectancy": round(trades['result'].mean(), 4)
            }
            results.append(res)
            print(f"    Result: {res}")
            
    pd.DataFrame(results).to_csv(out_dir / "phase10_family_screening_results.csv", index=False)
    print("Screening Complete.")

if __name__ == "__main__":
    run_screening()



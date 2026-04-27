
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import timedelta

class Phase10DiscoveryEngine:
    def __init__(self):
        self.tz_ny = 'America/New_York'

    def get_fractals(self, df, n=8):
        df = df.copy()
        df['is_high_fractal'] = (df['high'] == df['high'].rolling(window=2*n+1, center=True).max())
        df['is_low_fractal'] = (df['low'] == df['low'].rolling(window=2*n+1, center=True).min())
        return df['is_high_fractal'].fillna(False), df['is_low_fractal'].fillna(False)

    def run_family_screening(self, df_m5, df_h1, news_df, family_id, config):
        """
        family_id: 1..8
        config: specific params
        """
        trades = []
        df_m5 = df_m5.copy()
        df_m5['timestamp_ny'] = df_m5['timestamp'].dt.tz_convert(self.tz_ny)
        df_h1['timestamp_ny'] = df_h1['timestamp'].dt.tz_convert(self.tz_ny)
        
        # EMA H1 for Context
        df_h1['ema50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
        df_h1['ema_slope'] = df_h1['ema50'].diff()
        
        # Prepare H1 levels (PDH/L)
        df_h1['date'] = df_h1['timestamp_ny'].dt.date
        daily_levels = df_h1.groupby('date').agg({'high': 'max', 'low': 'min'}).shift(1)
        
        # Merge H1 context into M5
        df_h1_sync = df_h1[['timestamp_ny', 'ema50', 'ema_slope']].rename(columns={'timestamp_ny': 'h1_time'})
        df_m5 = pd.merge_asof(df_m5.sort_values('timestamp_ny'), df_h1_sync.sort_values('h1_time'), 
                             left_on='timestamp_ny', right_on='h1_time', direction='backward')
        
        # News mapping
        news_df['timestamp_utc'] = pd.to_datetime(news_df['timestamp_utc'], utc=True)
        news_times = news_df['timestamp_utc'].tolist()
        
        last_trade_date = None
        
        # Simulation Loop
        i = 0
        while i < len(df_m5):
            row = df_m5.iloc[i]
            curr_time = row.timestamp_ny
            curr_date = curr_time.date()
            
            # Constraints
            if curr_time.hour < 7 or curr_time.hour >= 20: 
                i += 1
                continue
            if curr_time.hour == 17 or curr_time.hour == 18:
                i += 1
                continue
            if config.get('one_trade_per_day') and last_trade_date == curr_date:
                i += 1
                continue
                
            # Family Logic
            setup = None
            
            if family_id == 1: # H1 Direction + LTF Pullback
                # Bias: EMA Slope + Price location
                bias = 0
                if row.ema_slope > 0 and row.close > row.ema50: bias = 1 # Long
                elif row.ema_slope < 0 and row.close < row.ema50: bias = -1 # Short
                
                if bias != 0:
                    # Pullback to M5 EMA 20 (simple proxy)
                    ema20_m5 = df_m5['close'].iloc[max(0, i-20):i].mean()
                    if bias == 1 and row.low <= ema20_m5 and row.close > ema20_m5:
                        setup = {'direction': 'LONG', 'entry_p': row.close, 'sl': row.low - 0.0001, 'tp': row.close + (row.close - (row.low - 0.0001)) * config.get('tp_r', 1.5)}
                    elif bias == -1 and row.high >= ema20_m5 and row.close < ema20_m5:
                        setup = {'direction': 'SHORT', 'entry_p': row.close, 'sl': row.high + 0.0001, 'tp': row.close - ((row.high + 0.0001) - row.close) * config.get('tp_r', 1.5)}

            if family_id == 4: # Opening Range Breakout (ORB)
                # Define OR: 08:00 - 09:30
                if curr_time.hour == 9 and curr_time.minute == 35:
                    or_data = df_m5[(df_m5['timestamp_ny'].dt.date == curr_date) & 
                                   (df_m5['timestamp_ny'].dt.hour >= 8) & 
                                   (df_m5['timestamp_ny'].dt.hour < 9) | 
                                   ((df_m5['timestamp_ny'].dt.hour == 9) & (df_m5['timestamp_ny'].dt.minute < 30))]
                    if not or_data.empty:
                        or_high = or_data['high'].max()
                        or_low = or_data['low'].min()
                        # Breakout
                        if row.close > or_high:
                            setup = {'direction': 'LONG', 'entry_p': row.close, 'sl': or_low, 'tp': row.close + (row.close - or_low) * 1.0}
                        elif row.close < or_low:
                            setup = {'direction': 'SHORT', 'entry_p': row.close, 'sl': or_high, 'tp': row.close - (or_high - row.close) * 1.0}

            # If setup found, process trade
            if setup:
                # News check
                in_news = any(abs((curr_time - nt).total_seconds()) < 30*60 for nt in news_times)
                if not in_news:
                    # Resolve trade
                    entry_p = setup['entry_p']
                    sl = setup['sl']
                    tp = setup['tp']
                    result = None
                    exit_time = None
                    
                    for j in range(i+1, min(i+100, len(df_m5))):
                        future = df_m5.iloc[j]
                        if setup['direction'] == 'LONG':
                            if future.low <= sl:
                                result = -1.0; exit_time = future.timestamp_ny; break
                            if future.high >= tp:
                                result = config.get('tp_r', 1.5); exit_time = future.timestamp_ny; break
                        else:
                            if future.high <= sl: # Wait, logic error for short SL? No, short SL is above entry.
                                pass 
                            # Fixing logic
                            if setup['direction'] == 'SHORT':
                                if future.high >= sl:
                                    result = -1.0; exit_time = future.timestamp_ny; break
                                if future.low <= tp:
                                    result = config.get('tp_r', 1.5); exit_time = future.timestamp_ny; break
                        if (future.timestamp_ny - curr_time).total_seconds() > 4*3600: # Timeout 4h
                            result = 0.0; exit_time = future.timestamp_ny; break
                    
                    if result is not None:
                        trades.append({'time': curr_time, 'direction': setup['direction'], 'result': result})
                        last_trade_date = curr_date
                        i = j # Skip to exit
                        continue

            i += 1
            
        return pd.DataFrame(trades)

def run_screening():
    print("Phase 10: Discovery Screening")
    engine = Phase10DiscoveryEngine()
    
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    
    # Load Data (M5 for screening speed)
    p = 'period_2020_2026'
    df_m5 = pd.read_csv(manifest[p]['m5_bid'])
    df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'], utc=True)
    df_h1 = pd.read_csv(manifest[p]['h1_bid'])
    df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True)
    news_df = pd.read_csv(manifest[p]['news'])
    
    families = [
        {"id": 1, "name": "H1_Trend_Pullback", "config": {"tp_r": 1.5, "one_trade_per_day": False}},
        {"id": 4, "name": "ORB_0800_0930", "config": {"tp_r": 1.0, "one_trade_per_day": True}}
    ]
    
    results = []
    for fam in families:
        print(f"  Screening Family: {fam['name']}...")
        trades = engine.run_family_screening(df_m5, df_h1, news_df, fam['id'], fam['config'])
        if not trades.empty:
            pf = trades[trades['result'] > 0]['result'].sum() / abs(trades[trades['result'] < 0]['result'].sum()) if any(trades['result'] < 0) else 1.0
            results.append({
                "family": fam['name'],
                "sample": len(trades),
                "pf": round(pf, 3),
                "expectancy": round(trades['result'].mean(), 4)
            })
            
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase10_high_frequency_entry_discovery\screening")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "phase10_family_screening_results.csv", index=False)
    print("Screening Complete.")

if __name__ == "__main__":
    run_screening()



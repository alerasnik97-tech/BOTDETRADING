
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pytz
from datetime import datetime, time

class Phase14Engine:
    def __init__(self, data_manifest_path):
        with open(data_manifest_path, 'r') as f:
            self.manifest = json.load(f)
        self.tz_ny = pytz.timezone("America/New_York")
        self.tz_utc = pytz.utc

    def load_and_prep_prices(self, period, timeframe='m5'):
        key_bid = f'{timeframe}_bid'
        key_ask = f'{timeframe}_ask'
        m_key_bid = key_bid if key_bid in self.manifest[period] else 'm5_bid'
        m_key_ask = key_ask if key_ask in self.manifest[period] else 'm5_ask'

        path_bid = self.manifest[period][m_key_bid]
        path_ask = self.manifest[period].get(m_key_ask)
        
        df_bid = pd.read_csv(path_bid)
        df_bid['timestamp'] = pd.to_datetime(df_bid['timestamp'], utc=True)
        
        if path_ask:
            df_ask = pd.read_csv(path_ask)
            df_ask['timestamp'] = pd.to_datetime(df_ask['timestamp'], utc=True)
            merged = pd.merge(df_bid, df_ask, on='timestamp', suffixes=('_bid', '_ask'))
        else:
            merged = df_bid.rename(columns={c: f"{c}_bid" for c in ['open', 'high', 'low', 'close'] if c in df_bid.columns})
            merged['open_ask'] = merged['open_bid'] + 0.0001
            merged['high_ask'] = merged['high_bid'] + 0.0001
            merged['low_ask'] = merged['low_bid'] + 0.0001
            merged['close_ask'] = merged['close_bid'] + 0.0001
            
        merged['timestamp_ny'] = merged['timestamp'].dt.tz_convert(self.tz_ny)
        
        if timeframe == 'm3':
            merged.set_index('timestamp', inplace=True)
            resampled = merged.resample('3min').agg({
                'open_bid': 'first', 'high_bid': 'max', 'low_bid': 'min', 'close_bid': 'last',
                'open_ask': 'first', 'high_ask': 'max', 'low_ask': 'min', 'close_ask': 'last'
            }).dropna().reset_index()
            resampled['timestamp_ny'] = resampled['timestamp'].dt.tz_convert(self.tz_ny)
            return resampled
            
        return merged

    def load_news(self, period):
        path = self.manifest[period]['news']
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp_utc'] if 'timestamp_utc' in df.columns else df['timestamp'], utc=True)
        return df

    def run_backtest(self, df_ltf, signals, news_df, config):
        """
        config: {
            "tp_r": 1.5, "be_r": None, "news_guard_mins": 30,
            "start_time": "07:00", "end_time": "20:00",
            "mandatory_close_time": "20:00", "one_trade_per_day": True
        }
        """
        trades = []
        active_trade = None
        news_timestamps = news_df['timestamp'].tolist()
        
        start_t = datetime.strptime(config['start_time'], "%H:%M").time()
        end_t = datetime.strptime(config['end_time'], "%H:%M").time()
        close_t = datetime.strptime(config['mandatory_close_time'], "%H:%M").time()
        
        signals_by_idx = {s['index']: s for s in signals}
        trades_today = 0
        current_date = None

        for i, row in df_ltf.iterrows():
            ny_time = row['timestamp_ny'].time()
            ny_date = row['timestamp_ny'].date()
            
            if ny_date != current_date:
                current_date = ny_date
                trades_today = 0

            if active_trade:
                # Mandatory close at 20:00 NY
                if ny_time >= close_t:
                    active_trade['status'] = 'TIMEOUT_CLOSE'
                    active_trade['exit_price'] = row['close_bid'] if active_trade['type'] == 'LONG' else row['close_ask']
                    active_trade['exit_time'] = row['timestamp_ny']
                    trades.append(active_trade)
                    active_trade = None
                    continue

                # Normal Exit Logic
                if active_trade['type'] == 'LONG':
                    if row['low_bid'] <= active_trade['sl']:
                        active_trade['status'] = 'SL'
                        active_trade['exit_price'] = active_trade['sl']
                        active_trade['exit_time'] = row['timestamp_ny']
                        trades.append(active_trade)
                        active_trade = None
                    elif row['high_bid'] >= active_trade['tp']:
                        active_trade['status'] = 'TP'
                        active_trade['exit_price'] = active_trade['tp']
                        active_trade['exit_time'] = row['timestamp_ny']
                        trades.append(active_trade)
                        active_trade = None
                    elif config.get('be_r') and not active_trade.get('be_triggered'):
                        if row['high_bid'] >= active_trade['entry_price'] + active_trade['risk'] * config['be_r']:
                            active_trade['sl'] = active_trade['entry_price']
                            active_trade['be_triggered'] = True
                else: # SHORT
                    if row['high_ask'] >= active_trade['sl']:
                        active_trade['status'] = 'SL'
                        active_trade['exit_price'] = active_trade['sl']
                        active_trade['exit_time'] = row['timestamp_ny']
                        trades.append(active_trade)
                        active_trade = None
                    elif row['low_ask'] <= active_trade['tp']:
                        active_trade['status'] = 'TP'
                        active_trade['exit_price'] = active_trade['tp']
                        active_trade['exit_time'] = row['timestamp_ny']
                        trades.append(active_trade)
                        active_trade = None
                    elif config.get('be_r') and not active_trade.get('be_triggered'):
                        if row['low_bid'] <= active_trade['entry_price'] - active_trade['risk'] * config['be_r']:
                            active_trade['sl'] = active_trade['entry_price']
                            active_trade['be_triggered'] = True
                continue
            
            # Entry Logic
            if i in signals_by_idx:
                if config.get('one_trade_per_day') and trades_today >= 1: continue
                if not (start_t <= ny_time <= end_t): continue
                
                # News Guard
                if any(abs((row['timestamp'] - nt).total_seconds()) <= config['news_guard_mins'] * 60 for nt in news_timestamps):
                    continue
                
                sig = signals_by_idx[i]
                if sig['type'] == 'LONG':
                    entry_p = row['close_ask']
                    sl = sig.get('sl_custom') or (row['low_bid'] - 0.0001)
                    risk = entry_p - sl
                    if risk <= 0: continue
                    active_trade = {
                        'type': 'LONG', 'entry_time': row['timestamp_ny'], 'entry_price': entry_p,
                        'sl': sl, 'tp': entry_p + risk * config['tp_r'], 'risk': risk, 'status': 'OPEN'
                    }
                else:
                    entry_p = row['close_bid']
                    sl = sig.get('sl_custom') or (row['high_ask'] + 0.0001)
                    risk = sl - entry_p
                    if risk <= 0: continue
                    active_trade = {
                        'type': 'SHORT', 'entry_time': row['timestamp_ny'], 'entry_price': entry_p,
                        'sl': sl, 'tp': entry_p - risk * config['tp_r'], 'risk': risk, 'status': 'OPEN'
                    }
                trades_today += 1
                    
        return pd.DataFrame(trades)

    def calculate_metrics(self, df_trades):
        if df_trades.empty:
            return {"sample": 0, "pf": 0, "expectancy": 0}
        df = df_trades.copy()
        df['r_return'] = np.where(df['status'] == 'TP', (df['tp'] - df['entry_price']).abs() / df['risk'], 
                         np.where(df['status'] == 'SL', -1.0, 
                         (df['exit_price'] - df['entry_price']) / df['risk'] if df.iloc[0]['type'] == 'LONG' else (df['entry_price'] - df['exit_price']) / df['risk']))
        
        profits = df[df['r_return'] > 0]['r_return'].sum()
        losses = abs(df[df['r_return'] < 0]['r_return'].sum())
        
        return {
            "sample": len(df),
            "pf": round(profits / losses if losses > 0 else 0, 2),
            "expectancy": round(df['r_return'].mean(), 3),
            "win_rate": round(len(df[df['r_return'] > 0]) / len(df), 4),
            "total_r": round(df['r_return'].sum(), 2),
            "timeout_count": len(df[df['status'] == 'TIMEOUT_CLOSE'])
        }

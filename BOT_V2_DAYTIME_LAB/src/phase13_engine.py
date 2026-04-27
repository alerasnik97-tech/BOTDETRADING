
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pytz
from datetime import datetime, timedelta
from phase13_signals import detect_h1_sweep_momentum, detect_session_reclaim
from phase13_helpers import get_all_levels

class Phase13Engine:
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
            # Fallback if ask not available, assume spread 0.0001
            merged = df_bid.copy()
            merged = merged.rename(columns={c: f"{c}_bid" for c in ['open', 'high', 'low', 'close'] if c in merged.columns})
            merged['open_ask'] = merged['open_bid'] + 0.0001
            merged['high_ask'] = merged['high_bid'] + 0.0001
            merged['low_ask'] = merged['low_bid'] + 0.0001
            merged['close_ask'] = merged['close_bid'] + 0.0001
            
        merged['timestamp_ny'] = merged['timestamp'].dt.tz_convert(self.tz_ny)
        
        if timeframe == 'm3':
            # Resample merged data
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
        if 'timestamp_utc' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp_utc'], utc=True)
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        return df

    def get_levels(self, df_h1):
        df = df_h1.copy()
        df['date'] = df['timestamp_ny'].dt.date
        daily = df.groupby('date').agg({'high_bid': 'max', 'low_bid': 'min'})
        daily['pdh'] = daily['high_bid'].shift(1)
        daily['pdl'] = daily['low_bid'].shift(1)
        return daily[['pdh', 'pdl']].to_dict('index')

    def run_backtest(self, df_ltf, levels, news_df, config):
        """
        config: {
            "method": "sweep" | "reclaim",
            "params": {...},
            "tp_r": 1.5,
            "sl_pips_plus": 0.5,
            "news_guard_mins": 30,
            "start_time": "08:00",
            "end_time": "12:00"
        }
        """
        if config['method'] == 'sweep':
            signals = detect_h1_sweep_momentum(df_ltf, levels, config['params'])
        else:
            signals = detect_session_reclaim(df_ltf, levels, config['params'])
            
        trades = []
        active_trade = None
        
        # News preparation
        news_timestamps = news_df['timestamp'].tolist()
        
        start_t = datetime.strptime(config['start_time'], "%H:%M").time()
        end_t = datetime.strptime(config['end_time'], "%H:%M").time()
        
        # Sort signals for iteration
        signals_by_idx = {s['index']: s for s in signals}
        
        trades_today = 0
        current_date = None

        for i, row in df_ltf.iterrows():
            if row['timestamp_ny'].date() != current_date:
                current_date = row['timestamp_ny'].date()
                trades_today = 0

            if active_trade:
                # Check for BE
                if not active_trade.get('be_triggered') and config.get('be_r'):
                    if active_trade['type'] == 'LONG':
                        if row['high_bid'] >= active_trade['entry_price'] + active_trade['risk'] * config['be_r']:
                            active_trade['sl'] = active_trade['entry_price']
                            active_trade['be_triggered'] = True
                    else:
                        if row['low_bid'] <= active_trade['entry_price'] - active_trade['risk'] * config['be_r']:
                            active_trade['sl'] = active_trade['entry_price']
                            active_trade['be_triggered'] = True

                # Exit logic
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
                continue
            
            # Entry logic
            if i in signals_by_idx:
                if config.get('one_trade_per_day') and trades_today >= 1: continue
                sig = signals_by_idx[i]
                if not (start_t <= row['timestamp_ny'].time() <= end_t): continue
                
                # News Filter
                ts_utc = row['timestamp']
                is_news = any(abs((ts_utc - nt).total_seconds()) <= config['news_guard_mins'] * 60 for nt in news_timestamps)
                if is_news: continue
                
                # M15 Bias Filter (Dual Bias)
                if config.get('use_m15_bias'):
                    m15_bias = config['m15_bias_map'].get(row['timestamp'])
                    if m15_bias is not None:
                        if sig['type'] == 'LONG' and m15_bias < 0: continue
                        if sig['type'] == 'SHORT' and m15_bias > 0: continue
                
                if sig['type'] == 'LONG':
                    entry_p = row['close_ask']
                    sl = sig.get('sl_custom') or (row['low_bid'] - config['sl_pips_plus'] * 0.0001)
                    risk = entry_p - sl
                    if risk <= 0: continue
                    active_trade = {
                        'type': 'LONG', 'entry_time': row['timestamp_ny'], 'entry_price': entry_p,
                        'sl': sl, 'tp': entry_p + risk * config['tp_r'], 'risk': risk, 'status': 'OPEN'
                    }
                    trades_today += 1
                else:
                    entry_p = row['close_bid']
                    sl = sig.get('sl_custom') or (row['high_ask'] + config['sl_pips_plus'] * 0.0001)
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
        df['r_return'] = np.where(df['status'] == 'TP', (df['tp'] - df['entry_price']).abs() / df['risk'], -1.0)
        
        profits = df[df['r_return'] > 0]['r_return'].sum()
        losses = abs(df[df['r_return'] < 0]['r_return'].sum())
        
        return {
            "sample": len(df),
            "pf": round(profits / losses if losses > 0 else 0, 2),
            "expectancy": round(df['r_return'].mean(), 3),
            "win_rate": round(len(df[df['r_return'] > 0]) / len(df), 4),
            "total_r": round(df['r_return'].sum(), 2)
        }

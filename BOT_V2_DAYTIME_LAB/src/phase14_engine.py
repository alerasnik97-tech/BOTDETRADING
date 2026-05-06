
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
        if isinstance(path_bid, dict): path_bid = path_bid['path']
        path_ask = self.manifest[period].get(m_key_ask)
        if isinstance(path_ask, dict): path_ask = path_ask['path']
        
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
        
        # Only resample if we are not already at the target timeframe
        # For M3, we check if the file name or manifest indicates it's already M3
        is_already_m3 = 'M3' in path_bid.upper()
        
        if timeframe == 'm3' and not is_already_m3:
            merged.set_index('timestamp', inplace=True)
            resampled = merged.resample('3min', closed='left', label='right').agg({
                'open_bid': 'first', 'high_bid': 'max', 'low_bid': 'min', 'close_bid': 'last',
                'open_ask': 'first', 'high_ask': 'max', 'low_ask': 'min', 'close_ask': 'last'
            }).shift(1).dropna().reset_index()
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
            "mandatory_close_time": "20:00", "one_trade_per_day": True,
            "max_trades_per_day": 1,
            "sl_buffer_pips": 0.0,
            "partial_tp_r": None, "partial_pct": 0.0,
            "timeout_mins": None
        }
        """
        trades = []
        active_trade = None
        
        # Pre-calculate news blocks for speed
        news_blocked_times = set()
        if not news_df.empty:
            news_guard = config.get('news_guard_mins', 30)
            for nt in news_df['timestamp']:
                for m in range(-news_guard, news_guard + 1):
                    news_blocked_times.add((nt + pd.Timedelta(minutes=m)).replace(second=0, microsecond=0))
        
        start_t = datetime.strptime(config['start_time'], "%H:%M").time()
        end_t = datetime.strptime(config['end_time'], "%H:%M").time()
        close_t = datetime.strptime(config['mandatory_close_time'], "%H:%M").time()
        
        signals_by_idx = {s['index']: s for s in signals}
        trades_today = 0
        current_date = None

        for row in df_ltf.itertuples():
            i = row.Index
            ny_time = row.timestamp_ny.time()
            ny_date = row.timestamp_ny.date()
            
            if ny_date != current_date:
                current_date = ny_date
                trades_today = 0

            if active_trade:
                # Timeout Logic
                elapsed_mins = (row.timestamp_ny - active_trade['entry_time']).total_seconds() / 60
                if (config.get('timeout_mins') and elapsed_mins >= config['timeout_mins']) or (ny_time >= close_t):
                    active_trade['status'] = 'TIMEOUT_CLOSE' if ny_time < close_t else 'FORCED_CLOSE_2000'
                    active_trade['exit_price'] = row.close_bid if active_trade['type'] == 'LONG' else row.close_ask
                    active_trade['exit_time'] = row.timestamp_ny
                    trades.append(active_trade)
                    active_trade = None
                    continue

                # Normal Exit Logic
                if active_trade['type'] == 'LONG':
                    # SL Check
                    if row.low_bid <= active_trade['sl']:
                        active_trade['status'] = 'SL'
                        active_trade['exit_price'] = active_trade['sl']
                        active_trade['exit_time'] = row.timestamp_ny
                        trades.append(active_trade)
                        active_trade = None
                        continue
                    
                    # Partial TP Check
                    if config.get('partial_tp_r') and not active_trade.get('partial_taken'):
                        if row.high_bid >= active_trade['entry_price'] + active_trade['risk'] * config['partial_tp_r']:
                            active_trade['partial_taken'] = True
                            active_trade['partial_exit_price'] = active_trade['entry_price'] + active_trade['risk'] * config['partial_tp_r']
                            active_trade['partial_exit_time'] = row.timestamp_ny
                    
                    # Final TP Check
                    if row.high_bid >= active_trade['tp']:
                        active_trade['status'] = 'TP'
                        active_trade['exit_price'] = active_trade['tp']
                        active_trade['exit_time'] = row.timestamp_ny
                        trades.append(active_trade)
                        active_trade = None
                        continue
                        
                    # BE Check
                    if config.get('be_r') and not active_trade.get('be_triggered'):
                        if row.high_bid >= active_trade['entry_price'] + active_trade['risk'] * config['be_r']:
                            active_trade['sl'] = active_trade['entry_price']
                            active_trade['be_triggered'] = True
                else: # SHORT
                    # SL Check
                    if row.high_ask >= active_trade['sl']:
                        active_trade['status'] = 'SL'
                        active_trade['exit_price'] = active_trade['sl']
                        active_trade['exit_time'] = row.timestamp_ny
                        trades.append(active_trade)
                        active_trade = None
                        continue
                    
                    # Partial TP Check
                    if config.get('partial_tp_r') and not active_trade.get('partial_taken'):
                        if row.low_bid <= active_trade['entry_price'] - active_trade['risk'] * config['partial_tp_r']:
                            active_trade['partial_taken'] = True
                            active_trade['partial_exit_price'] = active_trade['entry_price'] - active_trade['risk'] * config['partial_tp_r']
                            active_trade['partial_exit_time'] = row.timestamp_ny

                    # Final TP Check
                    if row.low_ask <= active_trade['tp']:
                        active_trade['status'] = 'TP'
                        active_trade['exit_price'] = active_trade['tp']
                        active_trade['exit_time'] = row.timestamp_ny
                        trades.append(active_trade)
                        active_trade = None
                        continue
                        
                    # BE Check
                    if config.get('be_r') and not active_trade.get('be_triggered'):
                        if row.low_bid <= active_trade['entry_price'] - active_trade['risk'] * config['be_r']:
                            active_trade['sl'] = active_trade['entry_price']
                            active_trade['be_triggered'] = True
                continue
            
            # Entry Logic
            if i in signals_by_idx:
                max_trades = config.get('max_trades_per_day', 1)
                if trades_today >= max_trades: continue
                if not (start_t <= ny_time <= end_t): continue
                
                # News Guard
                if row.timestamp.replace(second=0, microsecond=0) in news_blocked_times:
                    continue
                
                sig = signals_by_idx[i]
                sl_buffer = config.get('sl_buffer_pips', 0.0) * 0.0001
                
                if sig['type'] == 'LONG':
                    entry_p = row.close_ask
                    sl = (sig.get('sl_custom') or (row.low_bid - 0.0001)) - sl_buffer
                    risk = entry_p - sl
                    if risk <= 0: continue
                    active_trade = {
                        'type': 'LONG', 'entry_time': row.timestamp_ny, 'entry_price': entry_p,
                        'sl': sl, 'tp': entry_p + risk * config['tp_r'], 'risk': risk, 'status': 'OPEN',
                        'partial_taken': False
                    }
                else:
                    entry_p = row.close_bid
                    sl = (sig.get('sl_custom') or (row.high_ask + 0.0001)) + sl_buffer
                    risk = sl - entry_p
                    if risk <= 0: continue
                    active_trade = {
                        'type': 'SHORT', 'entry_time': row.timestamp_ny, 'entry_price': entry_p,
                        'sl': sl, 'tp': entry_p - risk * config['tp_r'], 'risk': risk, 'status': 'OPEN',
                        'partial_taken': False
                    }
                trades_today += 1
                    
        return pd.DataFrame(trades)

    def calculate_metrics(self, df_trades, config=None):
        if df_trades.empty:
            return {"sample": 0, "pf": 0.0, "expectancy": 0.0, "win_rate": 0.0, "total_r": 0.0}
        
        df = df_trades.copy()
        partial_pct = config.get('partial_pct', 0.0) if config else 0.0
        
        def calc_r_return(row):
            # Final R-return calculation considering partials
            final_pnl_r = 0.0
            
            # Exit calculation
            exit_dist = (row['exit_price'] - row['entry_price']) if row['type'] == 'LONG' else (row['entry_price'] - row['exit_price'])
            exit_r = exit_dist / row['risk']
            
            if row.get('partial_taken'):
                partial_dist = (row['partial_exit_price'] - row['entry_price']) if row['type'] == 'LONG' else (row['entry_price'] - row['partial_exit_price'])
                partial_r = partial_dist / row['risk']
                final_pnl_r = (partial_r * partial_pct) + (exit_r * (1.0 - partial_pct))
            else:
                final_pnl_r = exit_r
                
            return final_pnl_r

        df['r_return'] = df.apply(calc_r_return, axis=1)
        
        profits = df[df['r_return'] > 0]['r_return'].sum()
        losses = abs(df[df['r_return'] < 0]['r_return'].sum())
        
        metrics = {
            "sample": len(df),
            "pf": round(profits / losses if losses > 0 else (profits if profits > 0 else 0.0), 2),
            "expectancy": round(df['r_return'].mean(), 3),
            "win_rate": round(len(df[df['r_return'] > 0]) / len(df), 4),
            "total_r": round(df['r_return'].sum(), 2),
            "timeout_count": len(df[df['status'].isin(['TIMEOUT_CLOSE', 'FORCED_CLOSE_2000'])]),
            "tp_count": len(df[df['status'] == 'TP']),
            "sl_count": len(df[df['status'] == 'SL']),
            "be_count": len(df[df['be_triggered'] == True]) if 'be_triggered' in df.columns else 0
        }
        return metrics

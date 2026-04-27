
import pandas as pd
import numpy as np
import json
import pytz
from datetime import datetime, timedelta
from pathlib import Path

class Phase12ResearchEngine:
    def __init__(self, manifest_path):
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        self.tz_ny = pytz.timezone("America/New_York")
        self.tz_utc = pytz.utc

    def load_data(self, period, timeframe='m5'):
        # Period should be 'period_2015_2019' or 'period_2020_2026'
        paths = self.manifest[period]
        bid_path = paths.get(f'{timeframe}_bid')
        ask_path = paths.get(f'{timeframe}_ask')
        
        if not bid_path or not ask_path:
            raise ValueError(f"Data for {timeframe} in {period} not found in manifest.")
            
        print(f"Loading {period} {timeframe} data...")
        df_bid = pd.read_csv(bid_path)
        df_ask = pd.read_csv(ask_path)
        
        df_bid['timestamp'] = pd.to_datetime(df_bid['timestamp'], utc=True)
        df_ask['timestamp'] = pd.to_datetime(df_ask['timestamp'], utc=True)
        
        merged = pd.merge(df_bid, df_ask, on='timestamp', suffixes=('_bid', '_ask'))
        merged['timestamp_ny'] = merged['timestamp'].dt.tz_convert(self.tz_ny)
        return merged

    def get_fractals(self, df, n=3):
        highs = df['high_bid'].values
        lows = df['low_bid'].values
        is_high = np.zeros(len(df), dtype=bool)
        is_low = np.zeros(len(df), dtype=bool)
        
        for i in range(n, len(df) - n):
            if all(highs[i] > highs[i-n:i]) and all(highs[i] > highs[i+1:i+n+1]):
                is_high[i] = True
            if all(lows[i] < lows[i-n:i]) and all(lows[i] < lows[i+1:i+n+1]):
                is_low[i] = True
        return is_high, is_low

    def get_atr(self, df, period=14):
        # row dicts don't have rolling, need to do it on df
        high = df['high_bid']
        low = df['low_bid']
        close = df['close_bid'].shift(1)
        tr = pd.concat([high - low, abs(high - close), abs(low - close)], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def load_news(self):
        news_list = []
        for p in self.manifest:
            if 'news' in self.manifest[p]:
                df = pd.read_csv(self.manifest[p]['news'])
                col = 'timestamp_utc' if 'timestamp_utc' in df.columns else 'timestamp'
                df['timestamp'] = pd.to_datetime(df[col], utc=True)
                news_list.append(df)
        return pd.concat(news_list).sort_values('timestamp')

    def get_h1_levels(self, df_h1):
        df = df_h1.copy()
        df['date'] = df['timestamp_ny'].dt.date
        daily = df.groupby('date').agg({'high_bid': 'max', 'low_bid': 'min'})
        daily['pdh'] = daily['high_bid'].shift(1)
        daily['pdl'] = daily['low_bid'].shift(1)
        return daily[['pdh', 'pdl']].to_dict('index')

    def get_session_levels(self, df_ltf):
        df = df_ltf.copy()
        df['date'] = df['timestamp_ny'].dt.date
        df['hour'] = df['timestamp_ny'].dt.hour
        
        # Asia Range: 20:00 (prev) to 03:00 (curr)
        df['asia_session'] = ((df['hour'] >= 20) | (df['hour'] < 3))
        df['trading_day'] = np.where(df['hour'] >= 20, df['date'] + timedelta(days=1), df['date'])
        
        # London Range: 03:00 to 07:00
        df['london_session'] = ((df['hour'] >= 3) & (df['hour'] < 7))
        
        asia = df[df['asia_session']].groupby('trading_day').agg({'high_bid': 'max', 'low_bid': 'min'})
        london = df[df['london_session']].groupby('date').agg({'high_bid': 'max', 'low_bid': 'min'})
        
        res = {}
        all_dates = df['date'].unique()
        for d in all_dates:
            res[d] = {
                "asia_h": asia.at[d, 'high_bid'] if d in asia.index else None,
                "asia_l": asia.at[d, 'low_bid'] if d in asia.index else None,
                "london_h": london.at[d, 'high_bid'] if d in london.index else None,
                "london_l": london.at[d, 'low_bid'] if d in london.index else None
            }
        return res

    def run_simulation(self, df_ltf, levels, news_df, config):
        """
        config includes 'signal_func' which returns 1 (LONG), -1 (SHORT) or 0
        """
        trades = []
        df = df_ltf.copy()
        df['time'] = df['timestamp_ny'].dt.time
        df['date'] = df['timestamp_ny'].dt.date
        
        start_t = datetime.strptime(config['start_time'], "%H:%M").time()
        end_t = datetime.strptime(config['end_time'], "%H:%M").time()
        
        active_trade = None
        trades_today = 0
        current_date = None
        
        # News preparation
        news_df['timestamp_ny'] = news_df['timestamp'].dt.tz_convert(self.tz_ny)
        news_by_date = {d: group['timestamp_ny'].tolist() for d, group in news_df.groupby(news_df['timestamp_ny'].dt.date)}

        records = df.to_dict('records')
        for i in range(len(records)):
            row = records[i]
            if row['date'] != current_date:
                current_date = row['date']
                trades_today = 0
                news_today = news_by_date.get(current_date, [])
            
            if active_trade:
                # 1. Check BE
                if not active_trade.get('be_triggered') and config.get('be_r'):
                    if active_trade['type'] == 'LONG':
                        if row['high_bid'] >= active_trade['entry_price'] + (active_trade['risk'] * config['be_r']):
                            active_trade['sl'] = active_trade['entry_price']
                            active_trade['be_triggered'] = True
                    else:
                        if row['low_bid'] <= active_trade['entry_price'] - (active_trade['risk'] * config['be_r']):
                            active_trade['sl'] = active_trade['entry_price']
                            active_trade['be_triggered'] = True

                # 2. Check Partials
                if not active_trade.get('partial_triggered') and config.get('partial_r'):
                    if active_trade['type'] == 'LONG':
                        if row['high_bid'] >= active_trade['entry_price'] + (active_trade['risk'] * config['partial_r']):
                            active_trade['partial_triggered'] = True
                            # We record that it happened, logic for PnL in metrics
                    else:
                        if row['low_bid'] <= active_trade['entry_price'] - (active_trade['risk'] * config['partial_r']):
                            active_trade['partial_triggered'] = True

                # 3. Check Timeout
                if config.get('timeout_mins'):
                    if (row['timestamp_ny'] - active_trade['entry_time']).total_seconds() / 60 >= config['timeout_mins']:
                        active_trade['status'] = 'TIMEOUT'
                        active_trade['exit_time'] = row['timestamp_ny']
                        active_trade['exit_price'] = row['close_bid'] if active_trade['type'] == 'LONG' else row['close_ask']
                        trades.append(active_trade)
                        active_trade = None
                        continue

                # 4. Check SL/TP
                if active_trade['type'] == 'LONG':
                    if row['low_bid'] <= active_trade['sl']:
                        active_trade['status'] = 'BE' if active_trade.get('be_triggered') else 'SL'
                        active_trade['exit_time'] = row['timestamp_ny']
                        active_trade['exit_price'] = active_trade['sl']
                        trades.append(active_trade)
                        active_trade = None
                    elif row['high_bid'] >= active_trade['tp']:
                        active_trade['status'] = 'TP'
                        active_trade['exit_time'] = row['timestamp_ny']
                        active_trade['exit_price'] = active_trade['tp']
                        trades.append(active_trade)
                        active_trade = None
                else: # SHORT
                    if row['high_ask'] >= active_trade['sl']:
                        active_trade['status'] = 'BE' if active_trade.get('be_triggered') else 'SL'
                        active_trade['exit_time'] = row['timestamp_ny']
                        active_trade['exit_price'] = active_trade['sl']
                        trades.append(active_trade)
                        active_trade = None
                    elif row['low_ask'] <= active_trade['tp']:
                        active_trade['status'] = 'TP'
                        active_trade['exit_time'] = row['timestamp_ny']
                        active_trade['exit_price'] = active_trade['tp']
                        trades.append(active_trade)
                        active_trade = None
                continue

            # Signal Check
            if start_t <= row['time'] <= end_t and trades_today < config.get('max_trades_day', 1):
                # News Filter
                news_block = config.get('news_block_mins', 30)
                if any(abs((nt - row['timestamp_ny']).total_seconds() / 60) <= news_block for nt in news_today):
                    continue
                
                signal = config['signal_func'](row, levels.get(row['date']))
                if signal != 0:
                    entry_time = row['timestamp_ny']
                    entry_price = row['close_ask'] if signal == 1 else row['close_bid']
                    
                    # SL Logic
                    sl = 0
                    if signal == 1:
                        sl = row['low_bid'] - (config.get('sl_buffer_pips', 0.5) * 0.0001)
                        risk = entry_price - sl
                    else:
                        sl = row['high_bid'] + (config.get('sl_buffer_pips', 0.5) * 0.0001)
                        risk = sl - entry_price
                        
                    if risk > 0:
                        active_trade = {
                            "type": "LONG" if signal == 1 else "SHORT",
                            "entry_time": entry_time,
                            "entry_price": entry_price,
                            "sl": sl,
                            "tp": entry_price + (risk * config['tp_r']) if signal == 1 else entry_price - (risk * config['tp_r']),
                            "risk": risk,
                            "date": row['date']
                        }
                        trades_today += 1
                        
        return pd.DataFrame(trades)

    def calculate_metrics(self, df_trades, config):
        if df_trades.empty: return {"sample": 0, "pf": 0}
        
        df = df_trades.copy()
        tp_r = config['tp_r']
        partial_r = config.get('partial_r')
        partial_pct = config.get('partial_pct', 0.5)
        
        def calc_r(row):
            if row['status'] == 'TP':
                if row.get('partial_triggered') and partial_r:
                    return (partial_r * partial_pct) + (tp_r * (1 - partial_pct))
                return tp_r
            if row['status'] == 'SL': return -1.0
            if row['status'] == 'BE': return 0.0
            if row['status'] == 'TIMEOUT':
                # Approx R for timeout
                pips = (row['exit_price'] - row['entry_price']) if row['type'] == 'LONG' else (row['entry_price'] - row['exit_price'])
                return pips / (row['risk'] if row['risk'] > 0 else 0.0001)
            return 0
            
        df['r_return'] = df.apply(calc_r, axis=1)
        
        gp = df[df['r_return'] > 0]['r_return'].sum()
        gl = abs(df[df['r_return'] < 0]['r_return'].sum())
        pf = gp / gl if gl > 0 else 0
        
        return {
            "sample": len(df),
            "pf": round(pf, 2),
            "expectancy": round(df['r_return'].mean(), 3),
            "win_rate": round(len(df[df['status']=='TP'])/len(df), 3) if len(df)>0 else 0,
            "max_drawdown_r": round(self.calculate_max_dd(df['r_return']), 2)
        }

    @staticmethod
    def calculate_max_dd(returns):
        cum_r = returns.cumsum()
        peak = cum_r.expanding().max()
        dd = cum_r - peak
        return abs(dd.min())


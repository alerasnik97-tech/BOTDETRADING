
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import json

class Phase12AdvancedEngine:
    def __init__(self):
        self.tz_ny = pytz.timezone("America/New_York")
        self.tz_utc = pytz.utc

    def get_levels(self, df_h1):
        df = df_h1.copy()
        df['date'] = df['timestamp'].dt.date
        daily = df.groupby('date').agg({'high': 'max', 'low': 'min'})
        pdh = daily['high'].shift(1)
        pdl = daily['low'].shift(1)
        
        df['hour'] = df['timestamp'].dt.hour
        df['trading_day'] = np.where(df['hour'] >= 20, df['date'] + timedelta(days=1), df['date'])
        asia = df[((df['hour'] >= 20) | (df['hour'] < 3))].groupby('trading_day').agg({'high': 'max', 'low': 'min'})
        london = df[(df['hour'] >= 3) & (df['hour'] < 8)].groupby('date').agg({'high': 'max', 'low': 'min'})
        
        levels = pd.DataFrame(index=daily.index)
        levels['pdh'] = pdh
        levels['pdl'] = pdl
        
        df['tr'] = np.maximum(df['high'] - df['low'], 
                             np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                        abs(df['low'] - df['close'].shift(1))))
        df['atr_h1'] = df['tr'].rolling(14).mean().shift(1)
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean().shift(1)
        
        daily_indicators = df.groupby('date').agg({'atr_h1': 'mean', 'ema50': 'last'})
        levels = levels.merge(daily_indicators.rename(columns={'atr_h1': 'atr'}), left_on='date', right_index=True, how='left')
        levels = levels.merge(asia.rename(columns={'high': 'asia_h', 'low': 'asia_l'}), left_on='date', right_index=True, how='left')
        levels = levels.merge(london.rename(columns={'high': 'london_h', 'low': 'london_l'}), left_on='date', right_index=True, how='left')
        
        return levels

    def get_fractals(self, df, n=3):
        highs = df['high']
        lows = df['low']
        is_low = (lows == lows.rolling(window=2*n+1, center=True).min())
        is_high = (highs == highs.rolling(window=2*n+1, center=True).max())
        return is_high.fillna(False).values, is_low.fillna(False).values

    def precalculate_last_fractals(self, df, n=3):
        is_high = df['is_high_fractal'].values
        is_low = df['is_low_fractal'].values
        highs = df['high'].values
        lows = df['low'].values
        size = len(df)
        last_high_val = np.full(size, np.nan)
        last_low_val = np.full(size, np.nan)
        curr_h, curr_l = np.nan, np.nan
        for i in range(size):
            if i >= n:
                if is_high[i-n]: curr_h = highs[i-n]
                if is_low[i-n]: curr_l = lows[i-n]
            last_high_val[i] = curr_h
            last_low_val[i] = curr_l
        return last_high_val, last_low_val

    def resolve_trade(self, rows, start_idx, trade_config, direction, config):
        entry_p = trade_config['entry_p']
        sl_p = trade_config['sl']
        tp_p = trade_config['tp']
        risk = trade_config['risk']
        spread = 0.00005
        
        be_r = config.get('be_r')
        be_trigger = entry_p + (risk * be_r * (1 if direction == 'LONG' else -1)) if be_r else None
        
        partial_r = config.get('partial_r')
        partial_pct = config.get('partial_pct', 0.0)
        partial_trigger = entry_p + (risk * partial_r * (1 if direction == 'LONG' else -1)) if partial_r else None
        
        timeout_bars = config.get('timeout_bars', 300)
        
        is_be = False
        is_partial = False
        
        for i in range(start_idx, min(start_idx + timeout_bars, len(rows))):
            row = rows[i]
            h, l = row.high, row.low
            
            if direction == 'LONG':
                # BE
                if be_trigger and not is_be and h >= be_trigger:
                    is_be = True
                    sl_p = entry_p
                # Partial
                if partial_trigger and not is_partial and h >= partial_trigger:
                    is_partial = True
                # SL
                if l <= sl_p:
                    return ('BE' if is_be else 'SL'), sl_p, row.timestamp_ny, is_partial, i
                # TP
                if h >= tp_p:
                    return 'TP', tp_p, row.timestamp_ny, is_partial, i
            else: # SHORT
                ask_h = h + spread
                ask_l = l + spread
                if be_trigger and not is_be and ask_l <= be_trigger:
                    is_be = True
                    sl_p = entry_p
                if partial_trigger and not is_partial and ask_l <= partial_trigger:
                    is_partial = True
                if ask_h >= sl_p:
                    return ('BE' if is_be else 'SL'), sl_p, row.timestamp_ny, is_partial, i
                if ask_l <= tp_p:
                    return 'TP', tp_p, row.timestamp_ny, is_partial, i
            
            if row.timestamp_ny.hour >= 20:
                return 'TIMEOUT', row.close, row.timestamp_ny, is_partial, i
                
        return 'TIMEOUT', rows[min(start_idx + timeout_bars - 1, len(rows)-1)].close, rows[min(start_idx + timeout_bars - 1, len(rows)-1)].timestamp_ny, is_partial, min(start_idx + timeout_bars - 1, len(rows)-1)

    def run_backtest(self, df_ltf, levels, news_df, config):
        df = df_ltf.copy()
        fractal_n = config.get('fractal_n', 3)
        df['is_high_fractal'], df['is_low_fractal'] = self.get_fractals(df, n=fractal_n)
        df['last_high_f'], df['last_low_f'] = self.precalculate_last_fractals(df, n=fractal_n)
        
        trades = []
        pending_setup = None
        start_t = datetime.strptime(config['start_hour'], "%H:%M").time()
        end_t = datetime.strptime(config['end_hour'], "%H:%M").time()
        
        # News blocks
        news_blocked_times = set()
        if not news_df.empty:
            news_times = pd.to_datetime(news_df['timestamp_utc'] if 'timestamp_utc' in news_df.columns else news_df['timestamp'], utc=True).dt.tz_convert(self.tz_ny)
            for nt in news_times:
                for m in range(-config.get('news_block_mins', 30), config.get('news_block_mins', 30) + 1):
                    news_blocked_times.add((nt + timedelta(minutes=m)).replace(second=0, microsecond=0))

        rows = list(df.itertuples())
        last_trade_date = None
        last_processed_date = None
        sweep_counts_today = {}
        
        i = 0
        while i < len(rows):
            row = rows[i]
            curr_time = row.timestamp_ny
            curr_date = curr_time.date()
            
            if curr_date != last_processed_date:
                last_processed_date = curr_date
                sweep_counts_today = {}

            if config.get('one_trade_per_day') and last_trade_date == curr_date:
                i += 1
                continue

            if start_t <= curr_time.time() <= end_t:
                lvls = levels.loc[curr_date] if curr_date in levels.index else None
                if lvls is None: 
                    i += 1
                    continue
                
                targets = {'pdh': lvls['pdh'], 'pdl': lvls['pdl'], 'asia_h': lvls.get('asia_h', np.nan), 'asia_l': lvls.get('asia_l', np.nan)}
                
                if pending_setup is None:
                    for name, val in targets.items():
                        if pd.isna(val): continue
                        if ('h' in name and row.high > val) or ('l' in name and row.low < val):
                            pending_setup = {
                                'direction': 'SHORT' if 'h' in name else 'LONG',
                                'extreme': row.high if 'h' in name else row.low,
                                'time': curr_time,
                                'level_val': val
                            }
                            break
                elif pending_setup:
                    # Update extreme
                    if pending_setup['direction'] == 'SHORT':
                        pending_setup['extreme'] = max(pending_setup['extreme'], row.high)
                    else:
                        pending_setup['extreme'] = min(pending_setup['extreme'], row.low)
                    
                    # CHoCH Check
                    is_choch = False
                    if pending_setup['direction'] == 'SHORT' and not pd.isna(row.last_low_f) and row.close < row.last_low_f:
                        is_choch = True
                    elif pending_setup['direction'] == 'LONG' and not pd.isna(row.last_high_f) and row.close > row.last_high_f:
                        is_choch = True
                    
                    if is_choch:
                        if curr_time.replace(second=0, microsecond=0) in news_blocked_times:
                            pending_setup = None
                            i += 1
                            continue
                        
                        # Candle Quality
                        body_pct = abs(row.close - row.open) / (row.high - row.low) if (row.high - row.low) > 0 else 0
                        if config.get('min_body_pct') and body_pct < config['min_body_pct']:
                            i += 1
                            continue
                            
                        # Entry
                        spread = 0.00005
                        entry_p = row.close + (spread if pending_setup['direction'] == 'LONG' else 0)
                        sl = pending_setup['extreme'] + (config.get('sl_plus_pips', 0.5) * 0.0001 * (1 if pending_setup['direction'] == 'SHORT' else -1))
                        risk = abs(entry_p - sl)
                        tp = entry_p + (risk * config['tp_r'] * (-1 if pending_setup['direction'] == 'SHORT' else 1))
                        
                        res, exit_p, exit_t, was_partial, exit_idx = self.resolve_trade(rows, i+1, {'entry_p': entry_p, 'sl': sl, 'tp': tp, 'risk': risk}, pending_setup['direction'], config)
                        
                        trades.append({
                            'date': curr_date, 'entry_time': curr_time, 'direction': pending_setup['direction'],
                            'result': res, 'r_value': self.calculate_r(res, was_partial, config),
                            'tp_r': config['tp_r'], 'be_r': config.get('be_r'), 'partial_r': config.get('partial_r')
                        })
                        last_trade_date = curr_date
                        pending_setup = None
                        i = exit_idx
                        continue
            
            if pending_setup and curr_time > pending_setup['time'] + timedelta(minutes=120):
                pending_setup = None
            
            i += 1
                
        return pd.DataFrame(trades)

    def calculate_r(self, res, was_partial, config):
        tp_r = config['tp_r']
        partial_r = config.get('partial_r')
        partial_pct = config.get('partial_pct', 0.5)
        
        if res == 'TP':
            if was_partial and partial_r:
                return (partial_r * partial_pct) + (tp_r * (1 - partial_pct))
            return tp_r
        if res == 'SL': return -1.0
        if res == 'BE':
            if was_partial and partial_r:
                return (partial_r * partial_pct) # Exit rest at BE
            return 0.0
        return 0.0


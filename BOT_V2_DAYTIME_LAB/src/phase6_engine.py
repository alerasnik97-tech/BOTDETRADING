
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from pathlib import Path

class Phase6Engine:
    def __init__(self):
        self.tz_ny = pytz.timezone("America/New_York")
        self.tz_utc = pytz.utc


    def get_levels(self, df_h1):
        """
        Calculates PDH/L, PWH/L, PMH/L, Asia H/L, London H/L.
        df_h1 must have 'timestamp' in NY.
        """
        df = df_h1.copy()
        df['date'] = df['timestamp'].dt.date
        
        # Daily
        daily = df.groupby('date').agg({'high': 'max', 'low': 'min'})
        pdh = daily['high'].shift(1)
        pdl = daily['low'].shift(1)
        
        # Weekly
        df['week'] = df['timestamp'].dt.isocalendar().week
        df['year'] = df['timestamp'].dt.year
        weekly = df.groupby(['year', 'week']).agg({'high': 'max', 'low': 'min'})
        pwh = weekly['high'].shift(1)
        pwl = weekly['low'].shift(1)
        
        # Monthly
        df['month'] = df['timestamp'].dt.month
        monthly = df.groupby(['year', 'month']).agg({'high': 'max', 'low': 'min'})
        pmh = monthly['high'].shift(1)
        pml = monthly['low'].shift(1)
        
        # Asia (20:00 - 03:00 NY)
        df['hour'] = df['timestamp'].dt.hour
        # Assign to "Trading Day"
        df['trading_day'] = np.where(df['hour'] >= 20, df['date'] + timedelta(days=1), df['date'])
        asia = df[((df['hour'] >= 20) | (df['hour'] < 3))].groupby('trading_day').agg({'high': 'max', 'low': 'min'})
        
        # London (03:00 - 08:00 NY)
        london = df[(df['hour'] >= 3) & (df['hour'] < 8)].groupby('date').agg({'high': 'max', 'low': 'min'})
        
        # Merge all into a daily levels dataframe
        levels = pd.DataFrame(index=daily.index)
        levels['pdh'] = pdh
        levels['pdl'] = pdl
        
        # Weekly/Monthly need mapping back to date
        levels = levels.reset_index()
        levels['week'] = pd.to_datetime(levels['date']).dt.isocalendar().week
        levels['year'] = pd.to_datetime(levels['date']).dt.year
        levels['month'] = pd.to_datetime(levels['date']).dt.month
        
        # Add ATR & EMA Filter (H1)
        df['tr'] = np.maximum(df['high'] - df['low'], 
                             np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                        abs(df['low'] - df['close'].shift(1))))
        df['atr_h1'] = df['tr'].rolling(14).mean().shift(1)
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean().shift(1)
        
        daily_indicators = df.groupby('date').agg({'atr_h1': 'mean', 'ema50': 'last'})
        levels = levels.merge(daily_indicators.rename(columns={'atr_h1': 'atr'}), left_on='date', right_index=True, how='left')
        
        levels = levels.merge(weekly.rename(columns={'high': 'pwh', 'low': 'pwl'}), on=['year', 'week'], how='left')
        levels = levels.merge(monthly.rename(columns={'high': 'pmh', 'low': 'pml'}), on=['year', 'month'], how='left')
        levels = levels.merge(asia.rename(columns={'high': 'asia_h', 'low': 'asia_l'}), left_on='date', right_index=True, how='left')
        levels = levels.merge(london.rename(columns={'high': 'london_h', 'low': 'london_l'}), left_on='date', right_index=True, how='left')
        
        return levels.set_index('date')

    @staticmethod
    def get_fractals(df, n=3):
        """
        Detects fractals of size N using vectorized rolling operations.
        """
        highs = df['high']
        lows = df['low']
        
        # A fractal is a local min/max in a window of 2n+1
        is_low = (lows == lows.rolling(window=2*n+1, center=True).min())
        is_high = (highs == highs.rolling(window=2*n+1, center=True).max())
        
        return is_high.fillna(False).values, is_low.fillna(False).values

    def detect_choch(self, df, idx, direction, fractal_n=3):
        if idx < 10: return False, None
        subset = df.iloc[:idx]
        high_fractals = subset[subset['is_high_fractal']]
        low_fractals = subset[subset['is_low_fractal']]
        
        if direction == 'SHORT':
            if low_fractals.empty: return False, None
            last_low = low_fractals.iloc[-1]['low']
            if df.iloc[idx]['close'] < last_low:
                return True, last_low
        else:
            if high_fractals.empty: return False, None
            last_high = high_fractals.iloc[-1]['high']
            if df.iloc[idx]['close'] > last_high:
                return True, last_high
                
        return False, None

    def resolve_trade(self, df, start_idx, trade_config, direction):
        entry_p = trade_config['entry_p']
        sl_p = trade_config['sl']
        tp_p = trade_config['tp']
        risk = trade_config['risk']
        be_r = trade_config.get('be_r')
        be_trigger = entry_p + (risk * be_r * (1 if direction == 'LONG' else -1)) if be_r else None
        
        is_be = False
        
        for i in range(start_idx, min(start_idx + 300, len(df))):
            row = df.iloc[i]
            h, l = row['high'], row['low']
            
            if direction == 'LONG':
                if be_trigger and not is_be and h >= be_trigger:
                    is_be = True
                    sl_p = entry_p
                if l <= sl_p:
                    return 'SL' if not is_be else 'BE', sl_p, row['timestamp_ny']
                if h >= tp_p:
                    return 'TP', tp_p, row['timestamp_ny']
            else: # SHORT
                if be_trigger and not is_be and l <= be_trigger:
                    is_be = True
                    sl_p = entry_p
                if h >= sl_p:
                    return 'SL' if not is_be else 'BE', sl_p, row['timestamp_ny']
                if l <= tp_p:
                    return 'TP', tp_p, row['timestamp_ny']
            
            if row['timestamp_ny'].hour >= 20:
                return 'TIMEOUT', row['close'], row['timestamp_ny']
                
        return 'TIMEOUT', df.iloc[min(start_idx + 299, len(df)-1)]['close'], df.iloc[min(start_idx + 299, len(df)-1)]['timestamp_ny']

    def is_displacement(self, df, idx, threshold=1.5):
        if idx < 20: return False
        body = abs(df.iloc[idx]['close'] - df.iloc[idx]['open'])
        avg_body = df['close'].iloc[idx-20:idx].sub(df['open'].iloc[idx-20:idx]).abs().mean()
        return body > (avg_body * threshold)

    def has_fvg(self, df, idx, direction):
        if idx < 2: return False
        if direction == 'LONG':
            return df.iloc[idx-2]['high'] < df.iloc[idx]['low']
        else:
            return df.iloc[idx-2]['low'] > df.iloc[idx]['high']

    @staticmethod
    def precalculate_last_fractals(df, n=3):
        """
        Pre-calculates the value of the most recent fractal for each bar,
        respecting the confirmation delay of N bars to avoid lookahead.
        """
        is_high = df['is_high_fractal'].values
        is_low = df['is_low_fractal'].values
        highs = df['high'].values
        lows = df['low'].values
        size = len(df)
        
        last_high_val = np.full(size, np.nan)
        last_low_val = np.full(size, np.nan)
        
        curr_h = np.nan
        curr_l = np.nan
        
        for i in range(size):
            # A fractal at index 'i-n' is only confirmed at index 'i'
            if i >= n:
                if is_high[i-n]: curr_h = highs[i-n]
                if is_low[i-n]: curr_l = lows[i-n]
            
            last_high_val[i] = curr_h
            last_low_val[i] = curr_l
            
        return last_high_val, last_low_val

    def run_phase6_backtest(self, df_ltf, levels, news_df, config):
        df = df_ltf.copy()
        if 'is_high_fractal' not in df.columns:
            df['is_high_fractal'], df['is_low_fractal'] = self.get_fractals(df, n=config.get('fractal_n', 3))
        
        # Pre-calculate last fractals for O(1) access in loop with confirmation delay
        df['last_high_f'], df['last_low_f'] = self.precalculate_last_fractals(df, n=config.get('fractal_n', 3))
        
        trades = []
        pending_setup = None
        
        start_t = datetime.strptime(config['start_hour'], "%H:%M").time()
        end_t = datetime.strptime(config['end_hour'], "%H:%M").time()
        
        # Pre-calculate news blocks
        news_blocked_times = set()
        if not news_df.empty:
            if 'timestamp_utc' in news_df.columns:
                news_times = pd.to_datetime(news_df['timestamp_utc'], utc=True).dt.tz_convert(self.tz_ny)
            else:
                news_times = pd.to_datetime(news_df['timestamp'], utc=True).dt.tz_convert(self.tz_ny)
            
            for nt in news_times:
                for m in range(-config.get('news_block_mins', 30), config.get('news_block_mins', 30) + 1):
                    news_blocked_times.add((nt + timedelta(minutes=m)).replace(second=0, microsecond=0))

        # Use itertuples for speed
        print(f"    Starting backtest loop on {len(df)} bars...", flush=True)
        rows = list(df.itertuples())
        i = 0
        last_trade_date = None
        last_processed_date = None
        setup_today = False
        
        while i < len(rows):
            if i % 200000 == 0 and i > 0:
                print(f"      Processed {i} bars...", flush=True)
            row = rows[i]
            curr_time = row.timestamp_ny
            curr_date = curr_time.date()
            
            if curr_date != last_processed_date:
                last_processed_date = curr_date
                setup_today = False
                sweep_counts_today = {} # track per level

            # One trade per day limit
            if config.get('one_trade_per_day') and last_trade_date == curr_date:
                i += 1
                continue

            # Optimization: only check levels if in operational window
            if start_t <= curr_time.time() <= end_t:
                lvls = levels.loc[curr_date] if curr_date in levels.index else None
                if lvls is None: 
                    i += 1
                    continue
                
                targets = {
                    'pdh': lvls['pdh'], 'pdl': lvls['pdl'],
                    'pwh': lvls['pwh'], 'pwl': lvls['pwl'],
                    'asia_h': lvls.get('asia_h', np.nan), 'asia_l': lvls.get('asia_l', np.nan),
                    'london_h': lvls.get('london_h', np.nan), 'london_l': lvls.get('london_l', np.nan)
                }
                curr_atr = lvls.get('atr', 0)
                ema_h1 = lvls.get('ema50', 0)
                
                news_active = curr_time.replace(second=0, microsecond=0) in news_blocked_times

                if pending_setup is None and not (config.get('min_atr') and curr_atr < config['min_atr']):
                    # Trend Filter Check
                    if config.get('trend_filter'):
                        if ema_h1 > 0:
                            # Level sweep should be in direction of trend? 
                            # Or Entry should be in direction of trend?
                            # Standard: Only Short if Price < EMA (Bearish)
                            pass # We'll check this at entry time
                    # Filter: Only take first sweep of day if config says so
                    first_sweep_only = config.get('first_sweep_only')
                    
                    for name, val in targets.items():
                        if pd.isna(val): continue
                        sweep_count = sweep_counts_today.get(name, 0)
                        
                        if first_sweep_only and sweep_count > 0:
                            continue
                            
                        if 'h' in name:
                            if row.high > val:
                                sweep_counts_today[name] = sweep_count + 1
                                pending_setup = {
                                    'direction': 'SHORT', 'level_name': name, 'level_val': val, 
                                    'extreme': row.high, 'time': curr_time, 
                                    'sweep_num': sweep_counts_today[name]
                                }
                                break
                        else:
                            if row.low < val:
                                sweep_counts_today[name] = sweep_count + 1
                                pending_setup = {
                                    'direction': 'LONG', 'level_name': name, 'level_val': val, 
                                    'extreme': row.low, 'time': curr_time,
                                    'sweep_num': sweep_counts_today[name]
                                }
                                break
                
                elif pending_setup:
                    if pending_setup['direction'] == 'SHORT':
                        if 'sweep_high' not in pending_setup or row.high > pending_setup['sweep_high']: 
                            pending_setup['sweep_high'] = row.high
                    else:
                        if 'sweep_low' not in pending_setup or row.low < pending_setup['sweep_low']: 
                            pending_setup['sweep_low'] = row.low
                    
                    # CHoCH check (True Fractal Logic - Optimized O(1))
                    is_choch = False
                    trigger_lvl = None
                    
                    if pending_setup['direction'] == 'SHORT':
                        last_low = row.last_low_f
                        if not pd.isna(last_low) and row.close < last_low:
                            is_choch = True
                            trigger_lvl = last_low
                    else:
                        last_high = row.last_high_f
                        if not pd.isna(last_high) and row.close > last_high:
                            is_choch = True
                            trigger_lvl = last_high
                    
                    if is_choch:
                        if news_active:
                            pending_setup = None
                            i += 1
                            continue
                        
                        if config.get('trend_exhaustion'):
                            if pending_setup['direction'] == 'SHORT' and row.close < ema_h1:
                                pending_setup = None
                                i += 1
                                continue
                            if pending_setup['direction'] == 'LONG' and row.close > ema_h1:
                                pending_setup = None
                                i += 1
                                continue

                        entry_type = config.get('entry_type', 1)
                        # Phase 7 Filters (Quality)
                        candle_body_pct = abs(row.close - row.open) / (row.high - row.low) if (row.high - row.low) > 0 else 0
                        
                        # Apply CHoCH quality filters if present in config
                        if config.get('min_body_pct') and candle_body_pct < config['min_body_pct']:
                            i += 1
                            continue
                            
                        spread = 0.00005
                        entry_p = row.close + (spread if pending_setup['direction'] == 'LONG' else 0)
                        sl_pips = config.get('sl_plus_pips', 0.5)
                        
                        if config.get('sl_type') == 'sweep':
                            sl = pending_setup['extreme'] + (sl_pips * 0.0001 if pending_setup['direction'] == 'SHORT' else -sl_pips * 0.0001)
                        else:
                            sl = row.high if pending_setup['direction'] == 'SHORT' else row.low
                            sl += (sl_pips * 0.0001 if pending_setup['direction'] == 'SHORT' else -sl_pips * 0.0001)
                        
                        risk = abs(entry_p - sl)
                        if risk < 0.00005: risk = 0.0001
                        tp_val = config.get('tp_val', 2.0)
                        tp = entry_p + (risk * tp_val * (-1 if pending_setup['direction'] == 'SHORT' else 1))
                        
                        # Fast resolution with spread realism
                        res, exit_p, exit_t = self.resolve_trade_itertuples(rows, i+1, {'entry_p': entry_p, 'sl': sl, 'tp': tp, 'risk': risk, 'be_r': config.get('be_r'), 'spread': spread}, pending_setup['direction'])
                        
                        trades.append({
                            'date': curr_date, 
                            'entry_time': curr_time, 
                            'direction': pending_setup['direction'],
                            'level': pending_setup['level_name'], 
                            'entry_type': entry_type, 
                            'entry_p': entry_p,
                            'sl': sl,
                            'tp': tp,
                            'result': res, 
                            'exit_time': exit_t,
                            'r_value': tp_val if res == 'TP' else (-1.0 if res == 'SL' else 0.0),
                            'body_pct': candle_body_pct,
                            'dist_to_lvl': abs(row.close - pending_setup['level_val']) * 10000,
                            'time_post_sweep': (curr_time - pending_setup['time']).total_seconds() / 60,
                            'sweep_num': pending_setup.get('sweep_num', 1),
                            'max_depth_pips': abs(pending_setup['extreme'] - pending_setup['level_val']) * 10000
                        })
                        last_trade_date = curr_date
                        pending_setup = None
                        # Skip bars until exit_t or next day
                        # For simplicity, just continue
                
                if pending_setup and curr_time > pending_setup['time'] + timedelta(minutes=120):
                    pending_setup = None
            
            i += 1

        return pd.DataFrame(trades)

    def resolve_trade_itertuples(self, rows, start_idx, trade_config, direction):
        entry_p = trade_config['entry_p']
        sl_p = trade_config['sl']
        tp_p = trade_config['tp']
        risk = trade_config['risk']
        be_r = trade_config.get('be_r')
        spread = trade_config.get('spread', 0.00005)
        be_trigger = entry_p + (risk * be_r * (1 if direction == 'LONG' else -1)) if be_r else None
        is_be = False
        
        for i in range(start_idx, min(start_idx + 300, len(rows))):
            row = rows[i]
            h, l = row.high, row.low
            if direction == 'LONG':
                # Long exits at BID (h, l are BID)
                if be_trigger and not is_be and h >= be_trigger:
                    is_be = True
                    sl_p = entry_p
                if l <= sl_p: return 'SL' if not is_be else 'BE', sl_p, row.timestamp_ny
                if h >= tp_p: return 'TP', tp_p, row.timestamp_ny
            else: # SHORT
                # Short exits at ASK (ASK = BID + Spread)
                ask_h = h + spread
                ask_l = l + spread
                if be_trigger and not is_be and ask_l <= be_trigger:
                    is_be = True
                    sl_p = entry_p
                if ask_h >= sl_p: return 'SL' if not is_be else 'BE', sl_p, row.timestamp_ny
                if ask_l <= tp_p: return 'TP', tp_p, row.timestamp_ny
            if row.timestamp_ny.hour >= 20: return 'TIMEOUT', row.close, row.timestamp_ny
        return 'TIMEOUT', rows[min(start_idx + 299, len(rows)-1)].close, rows[min(start_idx + 299, len(rows)-1)].timestamp_ny





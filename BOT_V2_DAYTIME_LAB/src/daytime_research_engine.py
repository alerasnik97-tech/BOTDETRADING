
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pytz
from datetime import datetime, timedelta

class DaytimeResearchEngine:
    def __init__(self, data_manifest_path):
        with open(data_manifest_path, 'r') as f:
            self.manifest = json.load(f)
        self.tz_ny = pytz.timezone("America/New_York")
        self.tz_utc = pytz.utc

    def load_and_prep_prices(self, period, timeframe='m5'):
        key_bid = f'{timeframe}_bid'
        key_ask = f'{timeframe}_ask'
        
        if key_bid not in self.manifest[period]:
            # Try to resample from m5 if available
            print(f"Timeframe {timeframe} not found for {period}. Resampling from m5...")
            df_m5 = self.load_and_prep_prices(period, timeframe='m5')
            
            # Map timeframe string to pandas offset
            tf_map = {'m15': '15min', 'm3': '3min', 'm30': '30min', 'h1': '1h'}
            offset = tf_map.get(timeframe, timeframe)
            
            df_m5.set_index('timestamp', inplace=True)
            
            resampled = df_m5.resample(offset, closed='left', label='right').agg({
                'open_bid': 'first', 'high_bid': 'max', 'low_bid': 'min', 'close_bid': 'last', 'volume_bid': 'sum',
                'open_ask': 'first', 'high_ask': 'max', 'low_ask': 'min', 'close_ask': 'last', 'volume_ask': 'sum'
            }).shift(1).dropna()
            
            resampled.reset_index(inplace=True)
            resampled['timestamp_ny'] = resampled['timestamp'].dt.tz_convert(self.tz_ny)
            return resampled

        path_bid = self.manifest[period][key_bid]
        path_ask = self.manifest[period][key_ask]
        
        print(f"Loading {period} {timeframe} data...")
        df_bid = pd.read_csv(path_bid)
        df_ask = pd.read_csv(path_ask)
        
        df_bid['timestamp'] = pd.to_datetime(df_bid['timestamp'])
        df_ask['timestamp'] = pd.to_datetime(df_ask['timestamp'])
        
        # Ensure UTC if missing
        if df_bid['timestamp'].dt.tz is None:
            df_bid['timestamp'] = df_bid['timestamp'].dt.tz_localize(self.tz_utc)
        if df_ask['timestamp'].dt.tz is None:
            df_ask['timestamp'] = df_ask['timestamp'].dt.tz_localize(self.tz_utc)
            
        merged = pd.merge(df_bid, df_ask, on='timestamp', suffixes=('_bid', '_ask'))
        merged['timestamp_ny'] = merged['timestamp'].dt.tz_convert(self.tz_ny)
        return merged

    def load_news(self, period):
        path = self.manifest[period]['news']
        df = pd.read_csv(path)
        # Handle different news schemas
        if 'timestamp_utc' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp_utc'])
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(self.tz_utc)
            
        return df

    def get_levels(self, df_h1):
        # Calculate PDH/PDL
        # Convert to NY first to define "Days" correctly
        df = df_h1.copy()
        df['date'] = df['timestamp_ny'].dt.date
        
        daily = df.groupby('date').agg({
            'high_bid': 'max',
            'low_bid': 'min'
        })
        
        daily['pdh'] = daily['high_bid'].shift(1)
        daily['pdl'] = daily['low_bid'].shift(1)
        
        return daily[['pdh', 'pdl']].to_dict('index')

    def get_asia_levels(self, df_ltf):
        df = df_ltf.copy()
        df['date'] = df['timestamp_ny'].dt.date
        df['hour'] = df['timestamp_ny'].dt.hour
        
        # Asia Range: 20:00 (prev day) to 03:00 (curr day)
        # We can simplify by taking any bar between 20:00 and 03:00
        # and assigning it to the "Trading Day" (which is the current day)
        
        df['asia_session'] = ((df['hour'] >= 20) | (df['hour'] < 3))
        
        # Group by "Trading Day" (if hour >= 20, it's for next day)
        df['trading_day'] = np.where(df['hour'] >= 20, df['date'] + timedelta(days=1), df['date'])
        
        asia = df[df['asia_session']].groupby('trading_day').agg({
            'high_bid': 'max',
            'low_bid': 'min'
        })
        
        return asia.rename(columns={'high_bid': 'asia_h', 'low_bid': 'asia_l'}).to_dict('index')

    def get_ny_opening_range(self, df_ltf, start="07:00", end="09:30"):
        df = df_ltf.copy()
        df['time'] = df['timestamp_ny'].dt.time
        df['date'] = df['timestamp_ny'].dt.date
        
        start_t = datetime.strptime(start, "%H:%M").time()
        end_t = datetime.strptime(end, "%H:%M").time()
        
        ny_or = df[(df['time'] >= start_t) & (df['time'] < end_t)].groupby('date').agg({
            'high_bid': 'max',
            'low_bid': 'min'
        })
        
        return ny_or.rename(columns={'high_bid': 'or_h', 'low_bid': 'or_l'}).to_dict('index')

    def get_h1_bias(self, df_h1):
        # Simple Bias: If close > 20 EMA
        df = df_h1.copy()
        df['ema_20'] = df['close_bid'].ewm(span=20, adjust=False).mean()
        df['bias'] = np.where(df['close_bid'] > df['ema_20'], 1, -1)
        
        # Shift bias by 1 bar to avoid lookahead
        df['bias'] = df['bias'].shift(1)
        
        return df.set_index('timestamp')['bias'].to_dict()

    def run_simulation(self, df_ltf, levels, config):
        """
        config example:
        {
            "start_time": "07:00",
            "end_time": "20:30",
            "tp_r": 1.5,
            "sl_pips": 1.0,
            "news_filter_mins": 30
        }
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
        news_today = []

        # Optimization: Pre-sort news by date for faster lookup
        news_by_date = {}
        if config.get('news_filter_df') is not None:
            df_n = config['news_filter_df'].copy()
            df_n['date'] = df_n['timestamp_ny'].dt.date
            for d, group in df_n.groupby('date'):
                news_by_date[d] = group['timestamp_ny'].tolist()

        for i, row in df.iterrows():
            if row['date'] != current_date:
                current_date = row['date']
                trades_today = 0
                news_today = news_by_date.get(current_date, [])
                
            if active_trade:
                # Check for BE
                if not active_trade.get('be_triggered', False) and config.get('be_trigger_r'):
                    trigger_p = 0
                    if active_trade['type'] == 'LONG':
                        trigger_p = active_trade['entry_price'] + (active_trade['risk'] * config['be_trigger_r'])
                        if row['high_bid'] >= trigger_p:
                            active_trade['sl'] = active_trade['entry_price']
                            active_trade['be_triggered'] = True
                    else: # SHORT
                        trigger_p = active_trade['entry_price'] - (active_trade['risk'] * config['be_trigger_r'])
                        if row['low_bid'] <= trigger_p:
                            active_trade['sl'] = active_trade['entry_price']
                            active_trade['be_triggered'] = True

                # Check for TP/SL
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

            # Look for entry
            if start_t <= row['time'] <= end_t and trades_today < config.get('max_trades_day', 1):
                level_data = levels.get(row['date'])
                if not level_data:
                    continue
                
                entry_type = config.get('entry_type', 'candle_close')
                
                # Check for SHORT Setup (PDH or Asia H sweep)
                h_level = level_data.get('pdh')
                if level_data.get('asia_h') and (h_level is None or level_data['asia_h'] > h_level):
                    # For daytime NY, Asia H is often a better target than PDH if it's below it
                    # but let's just check both or the most relevant one.
                    # I'll just use Asia H if config says 'asia'
                    pass
                
                if config.get('use_asia'):
                    h_level = level_data.get('asia_h')
                    l_level = level_data.get('asia_l')
                else:
                    h_level = level_data.get('pdh')
                    l_level = level_data.get('pdl')

                if h_level is None or l_level is None: continue

                if config.get('post_news_only'):
                    # Check if there was a news event before current row today
                    if not any(nt < row['timestamp_ny'] for nt in news_today):
                        continue

                # SHORT Setup
                if row['high_bid'] > h_level:
                    confirmed = False
                    if entry_type == 'candle_close' and row['close_bid'] < level_data['pdh']:
                        confirmed = True
                    elif entry_type == 'engulfing' and self.is_engulfing(df, i, 'SHORT'):
                        confirmed = True
                    elif entry_type == 'fvg' and self.is_fvg(df, i, 'SHORT'):
                        confirmed = True
                    elif entry_type == 'choch' and self.is_choch(df, i, 'SHORT'):
                        confirmed = True
                        
                    if confirmed:
                        entry_p = row['close_bid']
                        sl = row['high_bid'] + (config['sl_pips'] * 0.0001)
                        risk = sl - entry_p
                        if risk > 0:
                            active_trade = {
                                "type": "SHORT",
                                "entry_time": row['timestamp_ny'],
                                "entry_price": entry_p,
                                "sl": sl,
                                "tp": entry_p - (risk * config['tp_r']),
                                "risk": risk,
                                "level": "H",
                                "entry_type": entry_type
                            }
                            trades_today += 1
                
                # LONG Setup
                elif row['low_bid'] < l_level:
                    confirmed = False
                    if entry_type == 'candle_close' and row['close_bid'] > level_data['pdl']:
                        confirmed = True
                    elif entry_type == 'engulfing' and self.is_engulfing(df, i, 'LONG'):
                        confirmed = True
                    elif entry_type == 'fvg' and self.is_fvg(df, i, 'LONG'):
                        confirmed = True
                    elif entry_type == 'choch' and self.is_choch(df, i, 'LONG'):
                        confirmed = True
                        
                    if confirmed:
                        entry_p = row['close_ask']
                        sl = row['low_bid'] - (config['sl_pips'] * 0.0001)
                        risk = entry_p - sl
                        if risk > 0:
                            active_trade = {
                                "type": "LONG",
                                "entry_time": row['timestamp_ny'],
                                "entry_price": entry_p,
                                "sl": sl,
                                "tp": entry_p + (risk * config['tp_r']),
                                "risk": risk,
                                "level": "L",
                                "entry_type": entry_type
                            }
                            trades_today += 1
                        
        return pd.DataFrame(trades)

    @staticmethod
    def is_engulfing(df, idx, side):
        if idx < 1: return False
        curr = df.iloc[idx]
        prev = df.iloc[idx-1]
        
        if side == 'LONG':
            return curr['close_bid'] > prev['open_bid'] and curr['open_bid'] < prev['close_bid'] and prev['close_bid'] < prev['open_bid']
        else: # SHORT
            return curr['close_bid'] < prev['open_bid'] and curr['open_bid'] > prev['close_bid'] and prev['close_bid'] > prev['open_bid']

    @staticmethod
    def is_fvg(df, idx, side):
        if idx < 2: return False
        p2 = df.iloc[idx-2]
        p1 = df.iloc[idx-1]
        curr = df.iloc[idx]
        
        if side == 'LONG':
            # Gap between p2 high and curr low
            return curr['low_bid'] > p2['high_bid']
        else: # SHORT
            # Gap between p2 low and curr high
            return curr['high_bid'] < p2['low_bid']

    @staticmethod
    def is_choch(df, idx, side, lookback=5):
        if idx < lookback: return False
        window = df.iloc[idx-lookback:idx]
        curr = df.iloc[idx]
        
        if side == 'LONG':
            # Break of a recent high
            recent_high = window['high_bid'].max()
            return curr['close_bid'] > recent_high
        else: # SHORT
            # Break of a recent low
            recent_low = window['low_bid'].min()
            return curr['close_bid'] < recent_low

def calculate_metrics(df_trades, tp_r=1.5):
    if df_trades.empty:
        return {"sample_size": 0, "pf": 0, "expectancy_r": 0}
    
    df = df_trades.copy()
    df['r_return'] = 0.0
    df.loc[df['status'] == 'TP', 'r_return'] = tp_r
    df.loc[df['status'] == 'SL', 'r_return'] = -1.0
    df.loc[df['status'] == 'BE', 'r_return'] = 0.0
    
    total_trades = len(df)
    tp_count = (df['status'] == 'TP').sum()
    sl_count = (df['status'] == 'SL').sum()
    be_count = (df['status'] == 'BE').sum()
    
    win_rate = tp_count / total_trades if total_trades > 0 else 0
    
    profits = df[df['r_return'] > 0]['r_return'].sum()
    losses = abs(df[df['r_return'] < 0]['r_return'].sum())
    
    metrics = {
        "sample_size": total_trades,
        "tp_count": int(tp_count),
        "sl_count": int(sl_count),
        "be_count": int(be_count),
        "win_rate": round(win_rate, 4),
        "cumulative_r": round(df['r_return'].sum(), 2),
        "pf": round(profits / losses if losses > 0 else 0, 2),
        "expectancy_r": round(df['r_return'].mean(), 2)
    }
    return metrics



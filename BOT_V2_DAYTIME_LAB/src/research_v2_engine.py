
import pandas as pd
import numpy as np
import os
import json
from datetime import time, datetime, timedelta
from pathlib import Path

class ResearchV5Engine:
    def __init__(self, manifest_path, news_path=None):
        with open(manifest_path, 'r') as f:
            self.paths = json.load(f)
        self.news = None
        if news_path and os.path.exists(news_path):
            self.news = pd.read_csv(news_path)
            self.news['timestamp_utc'] = pd.to_datetime(self.news['timestamp_utc'])

    def load_prices(self, period, timeframe='m5'):
        key = f"{timeframe}_bid"
        if key not in self.paths[period]: return None
        path = self.paths[period][key]
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('America/New_York')
        return df

    def get_levels(self, df_h1):
        df = df_h1.copy()
        df['date'] = df.index.date
        daily = df.resample('D', closed='left', label='right').agg({'high': 'max', 'low': 'min'}).shift(1)
        pdh, pdl = daily['high'].shift(1), daily['low'].shift(1)
        weekly = df.resample('W', closed='left', label='right').agg({'high': 'max', 'low': 'min'}).shift(1)
        pwh, pwl = weekly['high'].shift(1), weekly['low'].shift(1)
        monthly = df.resample('ME', closed='left', label='right').agg({'high': 'max', 'low': 'min'}).shift(1)
        pmh, pml = monthly['high'].shift(1), monthly['low'].shift(1)
        asia = df.between_time('18:00', '00:00').resample('D', closed='left', label='right').agg({'high': 'max', 'low': 'min'}).shift(1).shift(1)
        levels = pd.DataFrame(index=daily.index)
        levels['pdh'], levels['pdl'] = pdh, pdl
        levels['pwh'], levels['pwl'] = pwh.reindex(levels.index, method='ffill'), pwl.reindex(levels.index, method='ffill')
        levels['pmh'], levels['pml'] = pmh.reindex(levels.index, method='ffill'), pml.reindex(levels.index, method='ffill')
        levels['asia_h'], levels['asia_l'] = asia['high'], asia['low']
        levels = levels.ffill()
        levels.index = pd.to_datetime(levels.index).tz_localize(None)
        return levels

    def apply_entry_logic(self, df, family, config):
        l_h, l_l = df['pdh'], df['pdl']
        lt = config.get('level_type', 'pdh')
        if lt == 'asia': l_h, l_l = df['asia_h'], df['asia_l']
        elif lt == 'pwh': l_h, l_l = df['pwh'], df['pwl']
        
        # SFP + Displacement
        if family == 'SFP_DISPLACEMENT':
            is_sfp_h = (df['high'] > l_h) & (df['close'] < l_h)
            is_sfp_l = (df['low'] < l_l) & (df['close'] > l_l)
            body = (df['close'] - df['open']).abs()
            avg_body = body.rolling(20).mean()
            is_displaced = body > (avg_body * config.get('displacement_mult', 1.5))
            
            df['setup_s'] = (is_sfp_h.shift(2)) & (is_displaced.shift(1)) & (df['close'].shift(1) < df['open'].shift(1))
            df['setup_l'] = (is_sfp_l.shift(2)) & (is_displaced.shift(1)) & (df['close'].shift(1) > df['open'].shift(1))
            # SL for SFP: High of sweep candle (i-2)
            df['sl_price_s'] = df['high'].shift(2) + 0.00005
            df['sl_price_l'] = df['low'].shift(2) - 0.00005

        # FVG Simple
        elif family == 'FVG_SIMPLE':
            is_sweep_h = (df['high'] > l_h)
            is_sweep_l = (df['low'] < l_l)
            has_fvg_s = (df['low'].shift(2) > df['high']) & (is_sweep_h.shift(2))
            has_fvg_l = (df['high'].shift(2) < df['low']) & (is_sweep_l.shift(2))
            df['setup_s'], df['setup_l'] = has_fvg_s, has_fvg_l
            df['sl_price_s'] = df['high'].shift(2) + 0.00005
            df['sl_price_l'] = df['low'].shift(2) - 0.00005

        # CHOCH Simple
        elif family == 'CHOCH_SIMPLE':
            n = 3
            is_sweep_h = (df['high'] > l_h)
            is_sweep_l = (df['low'] < l_l)
            recent_low = df['low'].rolling(n).min().shift(1)
            recent_high = df['high'].rolling(n).max().shift(1)
            df['setup_s'] = (is_sweep_h.shift(n)) & (df['close'] < recent_low)
            df['setup_l'] = (is_sweep_l.shift(n)) & (df['close'] > recent_high)
            df['sl_price_s'] = df['high'].rolling(n).max().shift(1) + 0.00005
            df['sl_price_l'] = df['low'].rolling(n).min().shift(1) - 0.00005

        # Engulfing
        elif family == 'ENGULFING':
            is_sweep_h = (df['high'] > l_h)
            is_sweep_l = (df['low'] < l_l)
            is_eng_s = (df['close'] < df['low'].shift(1)) & (df['open'] >= df['close'].shift(1))
            is_eng_l = (df['close'] > df['high'].shift(1)) & (df['open'] <= df['close'].shift(1))
            df['setup_s'] = (is_sweep_h.shift(1)) & (is_eng_s)
            df['setup_l'] = (is_sweep_l.shift(1)) & (is_eng_l)
            df['sl_price_s'] = df['high'].shift(1) + 0.00005
            df['sl_price_l'] = df['low'].shift(1) - 0.00005

        # Reclaim
        elif family == 'RECLAIM':
            is_sweep_h = (df['high'] > l_h)
            is_sweep_l = (df['low'] < l_l)
            reclaim_h = (df['close'] < l_h) & (is_sweep_h)
            reclaim_l = (df['close'] > l_l) & (is_sweep_l)
            df['setup_s'] = (reclaim_h.shift(1)) & (df['close'] < df['low'].shift(1))
            df['setup_l'] = (reclaim_l.shift(1)) & (df['close'] > df['high'].shift(1))
            df['sl_price_s'] = df['high'].shift(1) + 0.00005
            df['sl_price_l'] = df['low'].shift(1) - 0.00005
            
        df['entry_signal'] = np.where(df.get('setup_s', False), -1, np.where(df.get('setup_l', False), 1, 0))
        return df

    def run_simulation(self, df_ltf, levels, config):
        family = config.get('family', 'SFP_DISPLACEMENT')
        df = df_ltf.copy()
        df['date'] = pd.to_datetime(df.index.date)
        df = df.join(levels, on='date')
        df = self.apply_entry_logic(df, family, config)
        df = df.between_time(config.get('start_time', '08:30'), config.get('end_time', '11:00'))
        
        trades = []
        last_date = None
        tp_r = config.get('tp_r', 2.0)
        be_at_r = config.get('be_at_r', None)
        spread = config.get('spread', 0.00005) # 0.5 pips realistic
        
        signals = df[df['entry_signal'] != 0]
        for idx, row in signals.iterrows():
            if last_date == idx.date(): continue
            
            direction = row['entry_signal']
            # Bid/Ask realism
            entry_p = row['close'] + spread if direction == 1 else row['close']
            sl_p = row['sl_price_s'] if direction == -1 else row['sl_price_l']
            
            risk = abs(entry_p - sl_p)
            if risk < 0.00005: risk = 0.0001
            tp_p = entry_p + (direction * risk * tp_r)
            be_trigger = entry_p + (direction * risk * (be_at_r or 1.0))
            
            future = df_ltf.loc[idx:]
            # We must process the entry candle itself first!
            res, r_val = "TIMEOUT", 0.0
            is_be = False
            
            for f_idx, f_row in future.iterrows():
                # Intra-candle simulation (simplified): 
                # If we are in the entry candle, check if it hit SL first or TP.
                # Since we don't have ticks, we assume worst case: SL if both are hit.
                
                h, l = f_row['high'], f_row['low']
                
                if direction == -1:
                    # Short: SL is high, TP is low
                    if h >= sl_p and l <= tp_p: # Both hit
                        res, r_val = "SL", (0.0 if is_be else -1.0)
                        break
                    elif h >= sl_p:
                        res, r_val = "SL", (0.0 if is_be else -1.0)
                        break
                    elif l <= tp_p:
                        res, r_val = "TP", tp_r
                        break
                    elif be_at_r and l <= be_trigger:
                        is_be = True
                        sl_p = entry_p # Move to BE
                else:
                    # Long: SL is low, TP is high
                    if l <= sl_p and h >= tp_p:
                        res, r_val = "SL", (0.0 if is_be else -1.0)
                        break
                    elif l <= sl_p:
                        res, r_val = "SL", (0.0 if is_be else -1.0)
                        break
                    elif h >= tp_p:
                        res, r_val = "TP", tp_r
                        break
                    elif be_at_r and h >= be_trigger:
                        is_be = True
                        sl_p = entry_p
                
                if f_idx > idx + timedelta(hours=6): break # Timeout 6h
            
            trades.append({'entry_time': idx, 'result': res, 'r_value': r_val})
            last_date = idx.date()
            
        return pd.DataFrame(trades)

def calculate_metrics(trades):
    if trades is None or trades.empty: return {"sample_size": 0, "pf": 0, "win_rate": 0, "expectancy": 0}
    tp = len(trades[trades['result'] == 'TP'])
    sl = len(trades[trades['result'] == 'SL'])
    total = tp + sl
    gp = trades[trades['r_value'] > 0]['r_value'].sum()
    gl = abs(trades[trades['r_value'] < 0]['r_value'].sum())
    return {
        "sample_size": total,
        "win_rate": round(tp/total if total > 0 else 0, 4),
        "pf": round(gp/gl if gl > 0 else 0, 4),
        "expectancy": round(trades['r_value'].mean(), 4)
    }



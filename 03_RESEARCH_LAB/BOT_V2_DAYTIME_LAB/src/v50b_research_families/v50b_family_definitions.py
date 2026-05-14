import pandas as pd
import numpy as np

class V50BBaseDetector:
    def __init__(self, config):
        self.config = config
        self.family_id = config.get("family_id", "FXX")

class F01LondonContinuation(V50BBaseDetector):
    def check_setup(self, df_m5, current_time, london_data):
        # london_data: {'dir': 1/-1, 'range_high': X, 'range_low': Y, 'ny_open_high': Z, 'ny_open_low': W}
        if not (9 <= current_time.hour < 11): return None
        if current_time.hour == 9 and current_time.minute < 30: return None
        
        dir = london_data['dir']
        last_price = df_m5['close'].iloc[-1]
        prev_price = df_m5['close'].iloc[-2]
        
        trigger_price = london_data['ny_open_high'] if dir == 1 else london_data['ny_open_low']
        
        if dir == 1 and prev_price <= trigger_price < last_price:
            return ('LONG', last_price, last_price - 0.0020, last_price + 0.0040)
        if dir == -1 and prev_price >= trigger_price > last_price:
            return ('SHORT', last_price, last_price + 0.0020, last_price - 0.0040)
        return None

class F06VolatilityRegime(V50BBaseDetector):
    def check_setup(self, df_m5, current_time, vol_data):
        # vol_data: {'is_compressed': True/False, 'range_high': X, 'range_low': Y}
        if not (8 <= current_time.hour < 11): return None
        if not vol_data['is_compressed']: return None
        
        last_price = df_m5['close'].iloc[-1]
        prev_price = df_m5['close'].iloc[-2]
        
        if prev_price <= vol_data['range_high'] < last_price:
            return ('LONG', last_price, last_price - 0.0015, last_price + 0.0030)
        if prev_price >= vol_data['range_low'] > last_price:
            return ('SHORT', last_price, last_price + 0.0015, last_price - 0.0030)
        return None

class F08SessionOverlap(V50BBaseDetector):
    def check_setup(self, df_m5, current_time):
        if not (8 <= current_time.hour < 12): return None
        sma20 = df_m5['close'].rolling(20).mean().iloc[-1]
        sma50 = df_m5['close'].rolling(50).mean().iloc[-1]
        last_price = df_m5['close'].iloc[-1]
        
        if pd.isna(sma20) or pd.isna(sma50): return None
        
        if sma20 > sma50 and last_price <= sma20: 
             return ('LONG', last_price, last_price - 0.0015, last_price + 0.0030)
        if sma20 < sma50 and last_price >= sma20: 
             return ('SHORT', last_price, last_price + 0.0015, last_price - 0.0030)
        return None

class F12MacroSafeWindow(V50BBaseDetector):
    def check_setup(self, df_m5, current_time, is_news_safe):
        if not (9 <= current_time.hour < 15): return None
        if not is_news_safe: return None
        # Inside Bar logic
        if len(df_m5) < 4: return None
        
        last_h = df_m5['high'].iloc[-1]
        last_l = df_m5['low'].iloc[-1]
        prev_h = df_m5['high'].iloc[-2]
        prev_l = df_m5['low'].iloc[-2]
        
        if last_h < prev_h and last_l > prev_l: # Inside Bar formed
            # We wait for breakout of the mother bar
            return ('LONG_STOP', prev_h + 0.0002, prev_l, prev_h + 0.0010)
        return None

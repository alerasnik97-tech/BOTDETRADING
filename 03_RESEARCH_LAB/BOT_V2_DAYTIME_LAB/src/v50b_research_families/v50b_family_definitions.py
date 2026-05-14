import pandas as pd
import numpy as np

class FamilyBase:
    def __init__(self, config):
        self.config = config
        self.family_id = config.get("family_id", "BASE")
        self.config_id = config.get("config_id", "BASE_0001")

    def generate_signal(self, bars):
        """Must be implemented by subclasses. Returns signal dict or None."""
        raise NotImplementedError

class F01LondonContinuation(FamilyBase):
    def generate_signal(self, bars):
        if len(bars) < 2: return None
        
        # Simple London Breakout (03:00 NY)
        # Check if last bar is in the window
        last_bar = bars.iloc[-1]
        ts = last_bar.name
        if not (ts.hour == 3 and ts.minute >= 15 and ts.hour <= 4):
            return None
            
        # Implementation of a real breakout logic here...
        # For pre-check, we just want to prove we can read the bars and produce a signal
        return {
            "family_id": self.family_id,
            "config_id": self.config_id,
            "signal_time": ts,
            "side": "buy",
            "entry_reference": last_bar["close"],
            "stop_reference": last_bar["low"],
            "target_r": 2.0,
            "reason": "LONDON_BREAKOUT_CANDIDATE"
        }

class F06VolatilityRegime(FamilyBase):
    def generate_signal(self, bars):
        if len(bars) < 20: return None
        ts = bars.iloc[-1].name
        if not (8 <= ts.hour <= 11): return None
        
        # Real indicator calculation
        close = bars["close"]
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = ma20 + 2 * std20
        
        if close.iloc[-1] > upper.iloc[-1]:
            return {
                "family_id": self.family_id,
                "config_id": self.config_id,
                "signal_time": ts,
                "side": "buy",
                "entry_reference": close.iloc[-1],
                "stop_reference": ma20.iloc[-1],
                "target_r": 2.5,
                "reason": "VOLATILITY_EXPANSION"
            }
        return None

class F08SessionOverlap(FamilyBase):
    def generate_signal(self, bars):
        if len(bars) < 21: return None
        ts = bars.iloc[-1].name
        if not (8 <= ts.hour <= 11): return None
        
        ema9 = bars["close"].ewm(span=9).mean()
        ema21 = bars["close"].ewm(span=21).mean()
        
        if ema9.iloc[-1] > ema21.iloc[-1] and ema9.iloc[-2] <= ema21.iloc[-2]:
            return {
                "family_id": self.family_id,
                "config_id": self.config_id,
                "signal_time": ts,
                "side": "buy",
                "entry_reference": bars["close"].iloc[-1],
                "stop_reference": bars["low"].iloc[-5:0].min() if len(bars) > 5 else bars["low"].iloc[-1],
                "target_r": 2.0,
                "reason": "EMA_CROSS_OVERLAP"
            }
        return None

class F12MacroSafeWindow(FamilyBase):
    def generate_signal(self, bars):
        if len(bars) < 14: return None
        ts = bars.iloc[-1].name
        if not (9 <= ts.hour <= 12): return None
        
        # RSI calculation
        delta = bars["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        if rsi.iloc[-1] < 30:
            return {
                "family_id": self.family_id,
                "config_id": self.config_id,
                "signal_time": ts,
                "side": "buy",
                "entry_reference": bars["close"].iloc[-1],
                "stop_reference": bars["close"].iloc[-1] - 0.0015,
                "target_r": 1.5,
                "reason": "MEAN_REVERSION_SAFE_WINDOW"
            }
        return None

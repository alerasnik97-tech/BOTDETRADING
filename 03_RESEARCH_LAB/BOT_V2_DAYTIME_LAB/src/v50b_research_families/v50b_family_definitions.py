import pandas as pd
import numpy as np

class FamilyBase:
    def __init__(self, config):
        self.config = config
        self.family_id = config.get("family_id", "BASE")
        self.config_id = config.get("config_id", "BASE_0001")
        self.target_r_override = config.get("target_r")

    def is_in_window(self, ts):
        window = self.config.get("session_window", "07:00-17:00")
        s_str, e_str = window.split("-")
        sh, sm = map(int, s_str.split(":"))
        eh, em = map(int, e_str.split(":"))
        # We assume ts is NY time for this check if called after conversion, 
        # but detectors often check hour directly. 
        # To be safe, we'll let the runner handle the timezone conversion and pass it.
        return (sh <= ts.hour < eh) or (ts.hour == eh and ts.minute <= em)

    def generate_signal(self, bars):
        """Must be implemented by subclasses. Returns signal dict or None."""
        raise NotImplementedError

class F01LondonContinuation(FamilyBase):
    def generate_signal(self, bars):
        # EXCLUDED FOR NOW
        return None

class F06VolatilityRegime(FamilyBase):
    def generate_signal(self, bars):
        lookback = int(self.config.get("realized_vol_lookback", 20))
        if len(bars) < lookback: return None
        ts = bars.iloc[-1].name # This is UTC
        
        # We check window in NY time (handled by runner usually, but let's be robust)
        # For simplicity in this specific lab, detectors check UTC or NY?
        # The runner will provide NY-localized bars to detectors.
        
        close = bars["close"]
        ma = close.rolling(lookback).mean()
        std = close.rolling(lookback).std()
        mult = float(self.config.get("bb_multiplier", 2.0))
        upper = ma + mult * std
        
        if close.iloc[-1] > upper.iloc[-1] and close.iloc[-2] <= upper.iloc[-2]:
            return {
                "family_id": self.family_id,
                "config_id": self.config_id,
                "signal_time": ts,
                "side": "buy",
                "entry_reference": close.iloc[-1],
                "stop_reference": ma.iloc[-1],
                "target_r": self.target_r_override or 2.5,
                "reason": "VOLATILITY_EXPANSION"
            }
        return None

class F08SessionOverlap(FamilyBase):
    def generate_signal(self, bars):
        fast = int(self.config.get("ema_fast", 9))
        slow = int(self.config.get("ema_slow", 21))
        if len(bars) < max(fast, slow): return None
        ts = bars.iloc[-1].name
        
        ema_f = bars["close"].ewm(span=fast).mean()
        ema_s = bars["close"].ewm(span=slow).mean()
        
        if ema_f.iloc[-1] > ema_s.iloc[-1] and ema_f.iloc[-2] <= ema_s.iloc[-2]:
            # Pullback depth check if provided
            return {
                "family_id": self.family_id,
                "config_id": self.config_id,
                "signal_time": ts,
                "side": "buy",
                "entry_reference": bars["close"].iloc[-1],
                "stop_reference": bars["low"].iloc[-5:].min(),
                "target_r": self.target_r_override or 2.0,
                "reason": "EMA_CROSS_OVERLAP"
            }
        return None

class F12MacroSafeWindow(FamilyBase):
    def generate_signal(self, bars):
        period = int(self.config.get("rsi_period", 14))
        if len(bars) < period + 1: return None
        ts = bars.iloc[-1].name
        
        delta = bars["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        os = float(self.config.get("rsi_oversold", 30))
        if rsi.iloc[-1] < os:
            return {
                "family_id": self.family_id,
                "config_id": self.config_id,
                "signal_time": ts,
                "side": "buy",
                "entry_reference": bars["close"].iloc[-1],
                "stop_reference": bars["close"].iloc[-1] - 0.0015,
                "target_r": self.target_r_override or 1.5,
                "reason": "RSI_OVERSOLD"
            }
        return None

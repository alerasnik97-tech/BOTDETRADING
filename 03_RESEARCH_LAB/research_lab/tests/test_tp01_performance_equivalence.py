import unittest
import numpy as np
import pandas as pd
from typing import Any

from research_lab.strategies.tp01_london_ny_momentum_pullback import (
    signal as optimized_signal,
    DEFAULT_PARAMS,
    _atr_series,
)

# Reference copy of the original O(N^2) signal logic for strict equivalence testing
def original_signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
    p = {**DEFAULT_PARAMS, **params}
    required = ("open", "high", "low", "close")
    if i <= 1 or not all(column in frame.columns for column in required):
        return None
    
    # Simple inline mock of _in_window to bypass session check if index is not timestamped
    # or just use standard entry_start/entry_end check if timestamped
    minute = frame.index[i].hour * 60 + frame.index[i].minute
    start_hour, start_min = map(int, str(p["entry_start"]).split(":"))
    end_hour, end_min = map(int, str(p["entry_end"]).split(":"))
    in_window = (start_hour * 60 + start_min) <= minute < (end_hour * 60 + end_min)
    if not in_window:
        return None

    atr_period = int(p["atr_period"])
    lookback = int(p["atr_percentile_lookback"])
    momentum_bars = int(p["momentum_bars"])
    if i < lookback + atr_period + momentum_bars:
        return None

    # Original ATR series call inside loop
    atr_values = _atr_series(frame, atr_period)
    current_atr = float(atr_values.iat[i])
    previous_atr_window = atr_values.iloc[i - lookback : i].dropna()
    if len(previous_atr_window) < lookback:
        return None
    threshold = float(np.percentile(previous_atr_window.to_numpy(dtype=float), float(p["atr_percentile"])))
    if not all(np.isfinite(val) for val in [current_atr, threshold]) or current_atr <= threshold or current_atr <= 0:
        return None

    # Original EWM calculation on iloc[:i]
    close_series = frame["close"].astype(float)
    ema = close_series.iloc[:i].ewm(span=int(p["ema_period"]), adjust=False).mean()
    if len(ema) < int(p["ema_period"]) + 2:
        return None
    ema_now = float(ema.iat[-1])
    ema_prev = float(ema.iat[-2])

    close = float(frame["close"].iat[i])
    prev_close = float(frame["close"].iat[i - 1])
    close_before_momentum = float(frame["close"].iat[i - momentum_bars - 1])
    momentum = prev_close - close_before_momentum
    required_momentum = float(p["momentum_atr_mult"]) * current_atr
    low = float(frame["low"].iat[i])
    high = float(frame["high"].iat[i])
    prev_high = float(frame["high"].iat[i - 1])
    prev_low = float(frame["low"].iat[i - 1])
    tolerance = float(p["pullback_tolerance_atr"]) * current_atr
    buffer = float(p["stop_atr_buffer"]) * current_atr
    
    if not all(np.isfinite(val) for val in [ema_now, ema_prev, close, prev_close, momentum, low, high, prev_high, prev_low]):
        return None

    long_bias = momentum > required_momentum and prev_close > ema_now and ema_now >= ema_prev
    short_bias = momentum < -required_momentum and prev_close < ema_now and ema_now <= ema_prev

    # Build signal inline to match build_signal exact outputs
    def build_signal_inline(direction: str, stop_price: float):
        if direction == "long":
            signal_value = 1
            if stop_price >= close:
                return None
        else:
            signal_value = -1
            if stop_price <= close:
                return None
        return {
            "signal": signal_value,
            "direction": direction,
            "stop_mode": "price",
            "stop_price": stop_price,
            "target_mode": "rr",
            "target_rr": float(p["target_rr"]),
            "break_even_at_r": None,
            "trailing_atr": False,
            "session_name": str(p["session_name"]),
        }

    if long_bias and low <= ema_now + tolerance and close > ema_now and close > prev_high:
        return build_signal_inline("long", min(low, prev_low) - buffer)
    if short_bias and high >= ema_now - tolerance and close < ema_now and close < prev_low:
        return build_signal_inline("short", max(high, prev_high) + buffer)
    return None


class TestTP01PerformanceEquivalence(unittest.TestCase):
    
    def _create_synthetic_frame(self, num_bars: int = 600) -> pd.DataFrame:
        np.random.seed(42)
        # Create continuous pricing
        close = 1.1000 + np.cumsum(np.random.normal(0, 0.0005, num_bars))
        high = close + np.random.uniform(0.0001, 0.0010, num_bars)
        low = close - np.random.uniform(0.0001, 0.0010, num_bars)
        open_val = (high + low) / 2
        
        # Generate timestamps strictly within 08:00-12:00 entry window (using seconds delta)
        timestamps = [pd.Timestamp("2023-01-01 08:00:00") + pd.Timedelta(seconds=idx) for idx in range(num_bars)]
        return pd.DataFrame({
            "open": open_val,
            "high": high,
            "low": low,
            "close": close
        }, index=timestamps)

    def test_tp01_signals_equivalent_before_after_on_synthetic_cases(self):
        # 1. Strict Equivalence Test across 600 bars (representing diverse market states)
        frame = self._create_synthetic_frame(600)
        
        # Run comparison
        for i in range(250, len(frame)):
            orig = original_signal(frame, i, {})
            opt = optimized_signal(frame, i, {})
            self.assertEqual(orig, opt, f"Mismatch at index {i}. Original: {orig}, Optimized: {opt}")

    def test_tp01_no_lookahead_shifted_features(self):
        # 2. No-Lookahead Test
        frame = self._create_synthetic_frame(500)
        i_target = 350
        
        # Calculate base signal
        signal_base = optimized_signal(frame, i_target, {})
        
        # Mutate future bars (e.g. at i_target + 1 or i_target + 5)
        frame_mutated = frame.copy()
        frame_mutated.loc[frame_mutated.index[i_target + 1] :, "close"] = 999.0
        frame_mutated.loc[frame_mutated.index[i_target + 1] :, "high"] = 999.0
        frame_mutated.loc[frame_mutated.index[i_target + 1] :, "low"] = -999.0
        
        # Recalculate signal on mutated frame (from scratch - clearing cache first)
        from research_lab.strategies.tp01_london_ny_momentum_pullback import _CACHE
        _CACHE.clear()
        
        signal_mutated = optimized_signal(frame_mutated, i_target, {})
        self.assertEqual(signal_base, signal_mutated, "Signal changed after future pricing mutation! Lookahead detected.")

    def test_tp01_repeated_signal_calls_do_not_recompute_full_history(self):
        # 3. Cache performance / reuse test
        frame = self._create_synthetic_frame(400)
        from research_lab.strategies.tp01_london_ny_momentum_pullback import _CACHE
        _CACHE.clear()
        
        # Call once to populate cache
        optimized_signal(frame, 300, {})
        self.assertEqual(len(_CACHE), 1, "Cache not populated!")
        
        # Fetch initial cache values
        key = list(_CACHE.keys())[0]
        atr_ref, ema_ref = _CACHE[key]
        
        # Call multiple times and assert identical object reference
        for idx in range(301, 310):
            optimized_signal(frame, idx, {})
            
        new_atr, new_ema = _CACHE[key]
        self.assertIs(atr_ref, new_atr, "ATR series was recomputed! Cache bypassed.")
        self.assertIs(ema_ref, new_ema, "EMA series was recomputed! Cache bypassed.")

    def test_tp01_performance_smoke(self):
        # 4. Performance Smoke Test (5000 bars processed in loop)
        frame = self._create_synthetic_frame(5000)
        from research_lab.strategies.tp01_london_ny_momentum_pullback import _CACHE
        _CACHE.clear()
        
        import time
        start_ts = time.time()
        for idx in range(300, len(frame)):
            optimized_signal(frame, idx, {})
        runtime = time.time() - start_ts
        print(f"\n[Smoke Performance] Processed {len(frame) - 300} bars in {runtime:.4f} seconds.")
        # Ensure it runs well under 1.0 second (typically runs in ~0.05-0.15s on standard CPU!)
        self.assertLess(runtime, 1.0, f"Performance regression! Runtime was {runtime:.4f}s.")

    def test_tp01_cache_fail_closed_or_invalidates(self):
        # 5. Cache invalidation on different frame size/contents
        frame1 = self._create_synthetic_frame(400)
        frame2 = self._create_synthetic_frame(500)
        
        from research_lab.strategies.tp01_london_ny_momentum_pullback import _CACHE
        _CACHE.clear()
        
        optimized_signal(frame1, 300, {})
        optimized_signal(frame2, 300, {})
        
        self.assertEqual(len(_CACHE), 2, "Cache failed to partition different frames!")

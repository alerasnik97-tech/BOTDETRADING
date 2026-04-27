from __future__ import annotations

import unittest

import pandas as pd

from research_lab.config import NY_TZ
from research_lab.strategies import pm_micro_reclaim_m3 as strategy


def make_strategy_frame(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(item["timestamp"], tz=NY_TZ) for item in rows])
    return pd.DataFrame(
        {
            "open": [item["open"] for item in rows],
            "high": [item["high"] for item in rows],
            "low": [item["low"] for item in rows],
            "close": [item["close"] for item in rows],
            "atr14": [item.get("atr14", 0.0005) for item in rows],
            "range_atr": [item.get("range_atr", 1.4) for item in rows],
            "h1_adx14": [item.get("h1_adx14", 18.0) for item in rows],
            "day_range_h1_atr": [item.get("day_range_h1_atr", 3.5) for item in rows],
            "h1_atr14": [item.get("h1_atr14", 0.0020) for item in rows],
            "h1_ema200": [item.get("h1_ema200", 1.1000) for item in rows],
            "vwap_dist_std": [item.get("vwap_dist_std", -2.0) for item in rows],
            "rsi2": [item.get("rsi2", 20.0) for item in rows],
        },
        index=index,
    )


class PmMicroReclaimStrategyTests(unittest.TestCase):
    def _make_prev_bar_sweep_reclaim_frame(self) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        base_time = pd.Timestamp("2024-01-03 11:00:00", tz=NY_TZ)
        for i in range(42):
            ts = base_time + pd.Timedelta(minutes=3 * i)
            rows.append(
                {
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": 1.1000,
                    "high": 1.1005,
                    "low": 1.0995,
                    "close": 1.1001,
                }
            )

        rows[-2].update(
            {
                "open": 1.1000,
                "high": 1.1002,
                "low": 1.0988,
                "close": 1.0991,
                "vwap_dist_std": -2.1,
                "rsi2": 19.0,
            }
        )
        rows[-1].update(
            {
                "open": 1.0992,
                "high": 1.1006,
                "low": 1.0990,
                "close": 1.1005,
                "vwap_dist_std": -2.0,
                "rsi2": 18.0,
            }
        )
        return make_strategy_frame(rows)

    def test_recent_sweep_allows_reclaim_one_bar_after_sweep(self) -> None:
        frame = self._make_prev_bar_sweep_reclaim_frame()

        params = {
            "session_name": "pm_11_16",
            "entry_minute_floor": 11 * 60,
            "latest_signal_minute": 16 * 60,
            "extremum_lookback": 12,
            "vwap_stretch_std": 1.8,
            "range_atr_min": 1.0,
            "range_atr_max": 2.8,
            "close_reclaim_min": 0.55,
            "stop_buffer_atr": 0.18,
            "target_rr": 1.2,
            "max_hold_bars": 5,
            "h1_adx_max": 24.0,
            "day_range_h1_atr_max": 5.0,
            "h1_ema_distance_max": 8.0,
            "rsi2_long_max": 28.0,
            "rsi2_short_min": 72.0,
            "cooldown_bars": 5,
            "allow_reclaim_after_sweep_bars": 1,
            "break_even_at_r": 0.8,
        }

        signal = strategy.signal(frame, len(frame) - 1, params)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["direction"], "long")
        self.assertAlmostEqual(float(signal["break_even_at_r"]), 0.8)

    def test_without_recent_sweep_allowance_same_pattern_does_not_trigger(self) -> None:
        frame = self._make_prev_bar_sweep_reclaim_frame()

        params = {
            "session_name": "pm_11_16",
            "entry_minute_floor": 11 * 60,
            "latest_signal_minute": 16 * 60,
            "extremum_lookback": 12,
            "vwap_stretch_std": 1.8,
            "range_atr_min": 1.0,
            "range_atr_max": 2.8,
            "close_reclaim_min": 0.55,
            "stop_buffer_atr": 0.18,
            "target_rr": 1.2,
            "max_hold_bars": 5,
            "h1_adx_max": 24.0,
            "day_range_h1_atr_max": 5.0,
            "h1_ema_distance_max": 8.0,
            "rsi2_long_max": 28.0,
            "rsi2_short_min": 72.0,
            "cooldown_bars": 5,
            "allow_reclaim_after_sweep_bars": 0,
        }

        signal = strategy.signal(frame, len(frame) - 1, params)
        self.assertIsNone(signal)

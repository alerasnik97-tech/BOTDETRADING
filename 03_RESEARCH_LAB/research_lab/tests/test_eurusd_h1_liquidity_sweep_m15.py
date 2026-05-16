from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research_lab.config import NY_TZ
from research_lab.strategies import eurusd_h1_liquidity_sweep_m15


class TestEurusdH1LiquiditySweepM15(unittest.TestCase):
    def setUp(self) -> None:
        times = pd.date_range("2024-01-03 10:00", periods=90, freq="15min", tz=NY_TZ)
        self.frame = pd.DataFrame(
            index=times,
            data={
                "open": 1.0990,
                "high": 1.0994,
                "low": 1.0988,
                "close": 1.0991,
                "atr14": 0.0008,
                "body_to_atr": 0.10,
                "body_fraction": 0.20,
                "close_location": 0.50,
                "range_vs_recent": 0.80,
                "prev_day_high": 1.1000,
                "prev_week_high": 1.1015,
            },
        )
        self.params = eurusd_h1_liquidity_sweep_m15.default_params()

    def test_emits_short_after_prev_day_high_sweep_and_m15_displacement(self) -> None:
        sweep_bar = pd.Timestamp("2024-01-03 12:00", tz=NY_TZ)
        confirm_bar = pd.Timestamp("2024-01-03 12:15", tz=NY_TZ)

        self.frame.at[sweep_bar, "open"] = 1.0998
        self.frame.at[sweep_bar, "high"] = 1.1003
        self.frame.at[sweep_bar, "low"] = 1.0994
        self.frame.at[sweep_bar, "close"] = 1.0998

        self.frame.at[confirm_bar, "open"] = 1.0999
        self.frame.at[confirm_bar, "high"] = 1.09995
        self.frame.at[confirm_bar, "low"] = 1.0990
        self.frame.at[confirm_bar, "close"] = 1.0992
        self.frame.at[confirm_bar, "body_to_atr"] = 0.9
        self.frame.at[confirm_bar, "body_fraction"] = 0.64
        self.frame.at[confirm_bar, "close_location"] = 0.18
        self.frame.at[confirm_bar, "range_vs_recent"] = 1.5

        idx = self.frame.index.get_loc(confirm_bar)
        signal = eurusd_h1_liquidity_sweep_m15.signal(self.frame, idx, self.params)

        self.assertIsNotNone(signal)
        self.assertEqual(signal["direction"], "short")
        self.assertAlmostEqual(signal["stop_price"], 1.10035)
        self.assertAlmostEqual(signal["target_rr"], 1.0)
        self.assertEqual(signal["max_hold_bars"], 8)

    def test_requires_close_through_sweep_low(self) -> None:
        sweep_bar = pd.Timestamp("2024-01-03 12:00", tz=NY_TZ)
        confirm_bar = pd.Timestamp("2024-01-03 12:15", tz=NY_TZ)

        self.frame.at[sweep_bar, "high"] = 1.1003
        self.frame.at[sweep_bar, "low"] = 1.0994
        self.frame.at[sweep_bar, "close"] = 1.0998

        self.frame.at[confirm_bar, "open"] = 1.0999
        self.frame.at[confirm_bar, "high"] = 1.09995
        self.frame.at[confirm_bar, "low"] = 1.0995
        self.frame.at[confirm_bar, "close"] = 1.0995
        self.frame.at[confirm_bar, "body_to_atr"] = 0.8
        self.frame.at[confirm_bar, "body_fraction"] = 0.60
        self.frame.at[confirm_bar, "close_location"] = 0.20
        self.frame.at[confirm_bar, "range_vs_recent"] = 1.4

        idx = self.frame.index.get_loc(confirm_bar)
        signal = eurusd_h1_liquidity_sweep_m15.signal(self.frame, idx, self.params)
        self.assertIsNone(signal)

    def test_requires_recent_sweep(self) -> None:
        confirm_bar = pd.Timestamp("2024-01-03 12:15", tz=NY_TZ)
        self.frame.at[confirm_bar, "open"] = 1.0999
        self.frame.at[confirm_bar, "high"] = 1.09995
        self.frame.at[confirm_bar, "low"] = 1.0990
        self.frame.at[confirm_bar, "close"] = 1.0992
        self.frame.at[confirm_bar, "body_to_atr"] = 0.9
        self.frame.at[confirm_bar, "body_fraction"] = 0.64
        self.frame.at[confirm_bar, "close_location"] = 0.18
        self.frame.at[confirm_bar, "range_vs_recent"] = 1.5
        self.frame["prev_day_high"] = np.nan

        idx = self.frame.index.get_loc(confirm_bar)
        signal = eurusd_h1_liquidity_sweep_m15.signal(self.frame, idx, self.params)
        self.assertIsNone(signal)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research_lab.config import NY_TZ
from research_lab.strategies import am_opening_drive_reversal


class TestAmOpeningDriveReversal(unittest.TestCase):
    def setUp(self) -> None:
        times = pd.date_range("2024-01-03 09:44", "2024-01-03 10:05", freq="1min", tz=NY_TZ)
        self.frame = pd.DataFrame(
            index=times,
            data={
                "open": 1.1000,
                "high": 1.1002,
                "low": 1.0998,
                "close": 1.1000,
                "atr14": 0.0004,
                "range_atr": 0.8,
                "odr_failure_short": False,
                "odr_failure_long": False,
                "odr_drive_high": 1.1012,
                "odr_drive_low": 1.0992,
                "odr_reclaim_level_short": 1.1003,
                "odr_reclaim_level_long": 1.0999,
            },
        )
        self.params = am_opening_drive_reversal.default_params()

    def test_emits_short_limit_after_bull_drive_failure(self) -> None:
        failure_bar = pd.Timestamp("2024-01-03 09:50", tz=NY_TZ)
        eval_bar = pd.Timestamp("2024-01-03 09:52", tz=NY_TZ)
        self.frame.at[failure_bar, "odr_failure_short"] = True
        self.frame.at[eval_bar, "close"] = 1.1000

        idx = self.frame.index.get_loc(eval_bar)
        signal = am_opening_drive_reversal.signal(self.frame, idx, self.params)

        self.assertIsNotNone(signal)
        self.assertEqual(signal["direction"], "short")
        self.assertEqual(signal["entry_mode"], "limit")
        self.assertAlmostEqual(signal["limit_price"], 1.1003)
        self.assertAlmostEqual(signal["stop_price"], 1.1013)
        self.assertAlmostEqual(signal["target_rr"], 1.5)

    def test_emits_long_limit_after_bear_drive_failure(self) -> None:
        failure_bar = pd.Timestamp("2024-01-03 09:51", tz=NY_TZ)
        eval_bar = pd.Timestamp("2024-01-03 09:53", tz=NY_TZ)
        self.frame["close"] = 1.1002
        self.frame["low"] = 1.1000
        self.frame.at[failure_bar, "odr_failure_long"] = True
        self.frame.at[eval_bar, "close"] = 1.1001

        idx = self.frame.index.get_loc(eval_bar)
        signal = am_opening_drive_reversal.signal(self.frame, idx, self.params)

        self.assertIsNotNone(signal)
        self.assertEqual(signal["direction"], "long")
        self.assertEqual(signal["entry_mode"], "limit")
        self.assertAlmostEqual(signal["limit_price"], 1.0999)
        self.assertAlmostEqual(signal["stop_price"], 1.0991)

    def test_does_not_reissue_after_reclaim_level_was_already_touched(self) -> None:
        failure_bar = pd.Timestamp("2024-01-03 09:50", tz=NY_TZ)
        touch_bar = pd.Timestamp("2024-01-03 09:51", tz=NY_TZ)
        eval_bar = pd.Timestamp("2024-01-03 09:52", tz=NY_TZ)
        self.frame.at[failure_bar, "odr_failure_short"] = True
        self.frame.at[touch_bar, "high"] = 1.1004
        self.frame.at[eval_bar, "close"] = 1.1000

        idx = self.frame.index.get_loc(eval_bar)
        signal = am_opening_drive_reversal.signal(self.frame, idx, self.params)
        self.assertIsNone(signal)

    def test_requires_failure_flag_not_just_trade_window(self) -> None:
        eval_bar = pd.Timestamp("2024-01-03 09:55", tz=NY_TZ)
        idx = self.frame.index.get_loc(eval_bar)
        signal = am_opening_drive_reversal.signal(self.frame, idx, self.params)
        self.assertIsNone(signal)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import pandas as pd

from research_lab.config import NY_TZ
from research_lab.strategies import am_opening_range_expansion_retest


class TestAmOpeningRangeExpansionRetest(unittest.TestCase):
    def setUp(self) -> None:
        times = pd.date_range("2024-01-03 09:46", "2024-01-03 10:05", freq="1min", tz=NY_TZ)
        self.frame = pd.DataFrame(
            index=times,
            data={
                "open": 1.1000,
                "high": 1.1004,
                "low": 1.1003,
                "close": 1.1003,
                "ore_ny_vwap": 1.1000,
                "ore_acceptance_long": False,
                "ore_acceptance_short": False,
                "ore_or_high": 1.1002,
                "ore_or_low": 1.0994,
                "ore_stop_price_long": 1.0997,
                "ore_stop_price_short": 1.0999,
            },
        )
        self.params = am_opening_range_expansion_retest.default_params()

    def test_emits_long_limit_after_acceptance(self) -> None:
        acceptance_bar = pd.Timestamp("2024-01-03 09:50", tz=NY_TZ)
        eval_bar = pd.Timestamp("2024-01-03 09:52", tz=NY_TZ)
        self.frame.at[acceptance_bar, "ore_acceptance_long"] = True
        self.frame.at[eval_bar, "close"] = 1.1005

        idx = self.frame.index.get_loc(eval_bar)
        signal = am_opening_range_expansion_retest.signal(self.frame, idx, self.params)

        self.assertIsNotNone(signal)
        self.assertEqual(signal["direction"], "long")
        self.assertEqual(signal["entry_mode"], "limit")
        self.assertAlmostEqual(signal["limit_price"], 1.1002)
        self.assertAlmostEqual(signal["stop_price"], 1.0997)
        self.assertAlmostEqual(signal["target_rr"], 1.5)

    def test_emits_short_limit_after_acceptance(self) -> None:
        acceptance_bar = pd.Timestamp("2024-01-03 09:51", tz=NY_TZ)
        eval_bar = pd.Timestamp("2024-01-03 09:53", tz=NY_TZ)
        self.frame["close"] = 1.0992
        self.frame["high"] = 1.0993
        self.frame["low"] = 1.0990
        self.frame["ore_ny_vwap"] = 1.0996
        self.frame.at[acceptance_bar, "ore_acceptance_short"] = True
        self.frame.at[eval_bar, "close"] = 1.0991

        idx = self.frame.index.get_loc(eval_bar)
        signal = am_opening_range_expansion_retest.signal(self.frame, idx, self.params)

        self.assertIsNotNone(signal)
        self.assertEqual(signal["direction"], "short")
        self.assertEqual(signal["entry_mode"], "limit")
        self.assertAlmostEqual(signal["limit_price"], 1.0994)
        self.assertAlmostEqual(signal["stop_price"], 1.0999)

    def test_does_not_reissue_after_retest_level_was_already_touched(self) -> None:
        acceptance_bar = pd.Timestamp("2024-01-03 09:50", tz=NY_TZ)
        touch_bar = pd.Timestamp("2024-01-03 09:51", tz=NY_TZ)
        eval_bar = pd.Timestamp("2024-01-03 09:52", tz=NY_TZ)
        self.frame.at[acceptance_bar, "ore_acceptance_long"] = True
        self.frame.at[touch_bar, "low"] = 1.1001
        self.frame.at[eval_bar, "close"] = 1.1005

        idx = self.frame.index.get_loc(eval_bar)
        signal = am_opening_range_expansion_retest.signal(self.frame, idx, self.params)
        self.assertIsNone(signal)

    def test_requires_acceptance_flag(self) -> None:
        eval_bar = pd.Timestamp("2024-01-03 09:55", tz=NY_TZ)
        idx = self.frame.index.get_loc(eval_bar)
        signal = am_opening_range_expansion_retest.signal(self.frame, idx, self.params)
        self.assertIsNone(signal)


if __name__ == "__main__":
    unittest.main()

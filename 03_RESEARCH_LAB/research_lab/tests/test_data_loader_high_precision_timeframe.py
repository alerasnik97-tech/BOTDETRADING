from __future__ import annotations

import unittest

import pandas as pd

from research_lab.config import DEFAULT_DATA_DIRS
from research_lab.data_loader import load_backtest_data_bundle


class TestHighPrecisionTargetTimeframe(unittest.TestCase):
    def test_high_precision_bundle_honors_requested_m5_timeframe(self) -> None:
        bundle = load_backtest_data_bundle(
            "EURUSD",
            list(DEFAULT_DATA_DIRS),
            "2024-10-01",
            "2024-10-07",
            "high_precision_mode",
            target_timeframe="M5",
        )
        self.assertTrue(bundle.frame.index.equals(bundle.precision_package["bid_exec"].index))
        self.assertTrue(bundle.frame.index.equals(bundle.precision_package["ask_exec"].index))
        self.assertEqual(bundle.frame.index[1] - bundle.frame.index[0], pd.Timedelta(minutes=5))


if __name__ == "__main__":
    unittest.main()

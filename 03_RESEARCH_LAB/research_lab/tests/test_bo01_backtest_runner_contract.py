# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from research_lab.runners import bo01_backtest_runner as runner

class TestBO01BacktestRunnerContract(unittest.TestCase):
    
    def test_import_and_constants(self):
        """Verify runner imports without side-effects and defines correct constants."""
        self.assertEqual(runner.RUNNER_ID, "BO01_BACKTEST_RUNNER_SYNTHETIC_V1")
        self.assertEqual(runner.STRATEGY_ID, "BO01")
        self.assertEqual(runner.ENTRY_POLICY, "ENTRY_NEXT_CANDLE_OPEN")
        self.assertEqual(runner.SAME_BAR_POLICY, "STOP_FIRST")
        self.assertEqual(runner.MAX_TRADES_PER_DAY, 1)
        self.assertEqual(runner.MAX_ACTIVE_POSITIONS, 1)

    def test_validate_backtest_frame_success(self):
        """Verify frame validation passes with a valid Datetime-indexed DataFrame."""
        idx = pd.date_range("2015-01-05 08:00:00", periods=5, freq="5min", tz="UTC")
        df = pd.DataFrame({
            "open": [1.1200, 1.1210, 1.1205, 1.1215, 1.1220],
            "high": [1.1215, 1.1225, 1.1210, 1.1230, 1.1225],
            "low": [1.1195, 1.1200, 1.1198, 1.1205, 1.1210],
            "close": [1.1210, 1.1205, 1.1215, 1.1220, 1.1215]
        }, index=idx)
        
        res = runner.validate_backtest_frame(df)
        self.assertTrue(res["ok"])
        self.assertEqual(res["row_count"], 5)
        self.assertIsNotNone(res["min_timestamp"])
        self.assertIsNotNone(res["max_timestamp"])

    def test_validate_backtest_frame_failures(self):
        """Verify frame validation fails closed with invalid inputs."""
        # 1. Non-DataFrame
        res = runner.validate_backtest_frame(None)
        self.assertFalse(res["ok"])
        self.assertIn("must be a non-None pandas DataFrame", "".join(res["errors"]))

        # 2. Empty DataFrame
        res = runner.validate_backtest_frame(pd.DataFrame())
        self.assertFalse(res["ok"])
        self.assertIn("DataFrame is empty", "".join(res["errors"]))

        # 3. Missing Column
        idx = pd.date_range("2015-01-05 08:00:00", periods=2, freq="5min", tz="UTC")
        df_missing = pd.DataFrame({"open": [1.1], "high": [1.2], "low": [0.9]}, index=idx[:1])
        res = runner.validate_backtest_frame(df_missing)
        self.assertFalse(res["ok"])
        self.assertIn("Missing required column", "".join(res["errors"]))

        # 4. Non-DatetimeIndex
        df_index = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}, index=[0])
        res = runner.validate_backtest_frame(df_index)
        self.assertFalse(res["ok"])
        self.assertIn("must be a pandas DatetimeIndex", "".join(res["errors"]))

        # 5. Timezone naive index
        idx_naive = pd.date_range("2015-01-05 08:00:00", periods=2, freq="5min")
        df_naive = pd.DataFrame({"open": [1.1, 1.2], "high": [1.3, 1.4], "low": [1.0, 1.1], "close": [1.2, 1.3]}, index=idx_naive)
        res = runner.validate_backtest_frame(df_naive)
        self.assertFalse(res["ok"])
        self.assertIn("must have an active timezone", "".join(res["errors"]))

        # 6. Monotonicity violation
        idx_unsorted = pd.DatetimeIndex([
            pd.Timestamp("2015-01-05 08:05:00", tz="UTC"),
            pd.Timestamp("2015-01-05 08:00:00", tz="UTC")
        ])
        df_unsorted = pd.DataFrame({"open": [1.1, 1.2], "high": [1.3, 1.4], "low": [1.0, 1.1], "close": [1.2, 1.3]}, index=idx_unsorted)
        res = runner.validate_backtest_frame(df_unsorted)
        self.assertFalse(res["ok"])
        self.assertIn("must be monotonically increasing", "".join(res["errors"]))

        # 7. NaNs in critical columns
        idx = pd.date_range("2015-01-05 08:00:00", periods=2, freq="5min", tz="UTC")
        df_nan = pd.DataFrame({"open": [1.1, float("nan")], "high": [1.3, 1.4], "low": [1.0, 1.1], "close": [1.2, 1.3]}, index=idx)
        res = runner.validate_backtest_frame(df_nan)
        self.assertFalse(res["ok"])
        self.assertIn("NaN value detected", "".join(res["errors"]))

    def test_validate_backtest_frame_unauthorized_dates(self):
        """Verify frame validation blocks access to 2025 and 2026."""
        idx_2025 = pd.date_range("2025-01-05 08:00:00", periods=2, freq="5min", tz="UTC")
        df_2025 = pd.DataFrame({"open": [1.1, 1.2], "high": [1.3, 1.4], "low": [1.0, 1.1], "close": [1.2, 1.3]}, index=idx_2025)
        res = runner.validate_backtest_frame(df_2025)
        self.assertFalse(res["ok"])
        self.assertIn("Unauthorized date detected: 2025", "".join(res["errors"]))

        idx_2026 = pd.date_range("2026-06-01 08:00:00", periods=2, freq="5min", tz="UTC")
        df_2026 = pd.DataFrame({"open": [1.1, 1.2], "high": [1.3, 1.4], "low": [1.0, 1.1], "close": [1.2, 1.3]}, index=idx_2026)
        res = runner.validate_backtest_frame(df_2026)
        self.assertFalse(res["ok"])
        self.assertIn("Unauthorized date detected: 2026", "".join(res["errors"]))

    def test_validate_backtest_frame_unauthorized_partitions(self):
        """Verify frame validation blocks access to validation and holdout splits."""
        idx = pd.date_range("2015-01-05 08:00:00", periods=2, freq="5min", tz="UTC")
        
        # 1. Validation split
        df_val = pd.DataFrame({
            "open": [1.1, 1.2], "high": [1.3, 1.4], "low": [1.0, 1.1], "close": [1.2, 1.3],
            "partition": ["train", "validation"]
        }, index=idx)
        res = runner.validate_backtest_frame(df_val)
        self.assertFalse(res["ok"])
        self.assertIn("Unauthorized data partition 'validation' detected", "".join(res["errors"]))

        # 2. Holdout split
        df_hold = pd.DataFrame({
            "open": [1.1, 1.2], "high": [1.3, 1.4], "low": [1.0, 1.1], "close": [1.2, 1.3],
            "dataset_split": ["holdout", "train"]
        }, index=idx)
        res = runner.validate_backtest_frame(df_hold)
        self.assertFalse(res["ok"])
        self.assertIn("Unauthorized data partition 'holdout' detected", "".join(res["errors"]))

    def test_validate_signal_contract_success(self):
        """Verify signal contract validation passes with a compliant dictionary."""
        sig = {
            "signal": 1,
            "direction": "long",
            "stop_price": 1.1150,
            "target_rr": 2.0
        }
        res = runner.validate_signal_contract(sig)
        self.assertTrue(res["ok"])

    def test_validate_signal_contract_failures(self):
        """Verify signal contract validation raises appropriate exceptions for malformed signals."""
        # 1. None signal type exception
        with self.assertRaises(TypeError):
            runner.validate_signal_contract(None)

        # 2. Non-dict signal type exception
        with self.assertRaises(TypeError):
            runner.validate_signal_contract([1, 2, 3])

        # 3. Missing keys raise ValueError
        with self.assertRaises(ValueError):
            runner.validate_signal_contract({"signal": 1, "direction": "long"})

        # 4. Invalid values raise ValueError
        with self.assertRaises(ValueError):
            runner.validate_signal_contract({
                "signal": 99,
                "direction": "long",
                "stop_price": 1.0,
                "target_rr": 2.0
            })

        with self.assertRaises(ValueError):
            runner.validate_signal_contract({
                "signal": 1,
                "direction": "long_side_invalid",
                "stop_price": 1.0,
                "target_rr": 2.0
            })

if __name__ == "__main__":
    unittest.main()

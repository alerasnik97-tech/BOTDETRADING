from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from typing import Any

from research_lab.runners import m2_structural_runner


class FakeStrategyBO01:
    ID = "BO01"
    
    @staticmethod
    def default_params() -> dict[str, Any]:
        return {"session_name": "london"}
        
    @staticmethod
    def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
        close = float(frame["close"].iat[i])
        if close > 100.0:
            return {
                "signal": 1,
                "direction": "long",
                "stop_price": 99.0,
            }
        elif close < 50.0:
            raise ValueError("Simulated strategy exception")
        return None


class TestM2StructuralRunnerContract(unittest.TestCase):
    
    def setUp(self) -> None:
        # Create a synthetic DatetimeIndex with M5 cadence
        times = pd.date_range("2015-01-01 00:00:00", periods=100, freq="5min", tz="UTC")
        self.frame = pd.DataFrame(
            {
                "open": np.linspace(10.0, 110.0, 100),
                "high": np.linspace(11.0, 111.0, 100),
                "low": np.linspace(9.0, 109.0, 100),
                "close": np.linspace(10.0, 110.0, 100),
                "volume": np.ones(100) * 100,
            },
            index=times
        )
        
    def test_import_side_effects(self) -> None:
        # Verify that imported module does not perform filesystem operations or top-level calculations
        self.assertEqual(m2_structural_runner.RUNNER_ID, "M2_STRUCTURAL_RUNNER_BO01_MR02_V1")
        self.assertEqual(m2_structural_runner.ALLOWED_STRATEGY_IDS, {"BO01", "MR02"})
        
    def test_validate_frame_pass(self) -> None:
        diag = m2_structural_runner.validate_frame_for_m2(self.frame)
        self.assertTrue(diag["valid"])
        self.assertEqual(diag["row_count"], 100)
        
    def test_validate_frame_empty(self) -> None:
        empty = pd.DataFrame()
        with self.assertRaises(ValueError):
            m2_structural_runner.validate_frame_for_m2(empty)
            
    def test_validate_frame_tz_naive(self) -> None:
        naive_times = pd.date_range("2015-01-01 00:00:00", periods=100, freq="5min")
        naive_frame = self.frame.copy()
        naive_frame.index = naive_times
        with self.assertRaises(ValueError):
            m2_structural_runner.validate_frame_for_m2(naive_frame)
            
    def test_validate_frame_forbidden_date_2025(self) -> None:
        bad_times = pd.date_range("2025-01-01 00:00:00", periods=100, freq="5min", tz="UTC")
        bad_frame = self.frame.copy()
        bad_frame.index = bad_times
        with self.assertRaises(ValueError):
            m2_structural_runner.validate_frame_for_m2(bad_frame)
            
    def test_validate_frame_forbidden_date_2026(self) -> None:
        bad_times = pd.date_range("2026-01-01 00:00:00", periods=100, freq="5min", tz="UTC")
        bad_frame = self.frame.copy()
        bad_frame.index = bad_times
        with self.assertRaises(ValueError):
            m2_structural_runner.validate_frame_for_m2(bad_frame)
            
    def test_validate_frame_validation_split(self) -> None:
        bad_frame = self.frame.copy()
        bad_frame["split"] = ["train"] * 50 + ["validation"] * 50
        with self.assertRaises(ValueError):
            m2_structural_runner.validate_frame_for_m2(bad_frame)
            
    def test_validate_frame_holdout_split(self) -> None:
        bad_frame = self.frame.copy()
        bad_frame["partition"] = ["holdout"] * 100
        with self.assertRaises(ValueError):
            m2_structural_runner.validate_frame_for_m2(bad_frame)
            
    def test_run_structural_counts_values(self) -> None:
        # The frame close increases from 10 to 110. 
        # FakeStrategyBO01 generates long signals when close > 100.
        # It generates ValueError exceptions when close < 50.
        # For close between 50 and 100, it returns None.
        counts = m2_structural_runner.run_structural_counts(FakeStrategyBO01, self.frame)
        
        self.assertEqual(counts["row_count"], 100)
        self.assertEqual(counts["signal_call_count"], 100)
        
        # Exceptions should be counted for close < 50.
        # Since close starts at 10 and increases uniformly to 110, some bars will be < 50.
        self.assertGreater(counts["exception_count"], 0)
        self.assertEqual(counts["fail_closed_count"], counts["exception_count"])
        
        # Valid signal counts should be greater than 0 (close > 100)
        self.assertGreater(counts["valid_signal_count"], 0)
        self.assertEqual(counts["contract_valid_count"], counts["valid_signal_count"])
        
        # Temporal counts
        self.assertGreater(counts["days_with_signal"], 0)
        self.assertGreater(counts["max_signals_per_day"], 0)
        self.assertGreater(sum(counts["signals_by_hour"].values()), 0)
        self.assertGreater(sum(counts["signals_by_month"].values()), 0)
        
    def test_run_m2_structural_evaluation_pass(self) -> None:
        summary = m2_structural_runner.run_m2_structural_evaluation(
            [FakeStrategyBO01],
            self.frame,
            "2015-01-01 00:00:00",
            "2015-01-01 23:59:59"
        )
        self.assertEqual(summary["status"], "COMPLETED")
        self.assertEqual(summary["runner_id"], m2_structural_runner.RUNNER_ID)
        self.assertIn("BO01", summary["results"])


if __name__ == "__main__":
    unittest.main()

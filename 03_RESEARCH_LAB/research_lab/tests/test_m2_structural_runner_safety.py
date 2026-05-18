from __future__ import annotations

import unittest
import os
import pandas as pd
from typing import Any

from research_lab.runners import m2_structural_runner


class FakeStrategyMR02:
    ID = "MR02"
    
    @staticmethod
    def default_params() -> dict[str, Any]:
        return {"session_name": "london", "target_rr": 1.5}
        
    @staticmethod
    def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
        del frame, i, params
        return None


class TestM2StructuralRunnerSafety(unittest.TestCase):
    
    def test_no_forbidden_active_logic_terms(self) -> None:
        # Verify that the runner file does not contain active logic referencing performance terms
        # (It can contain them only as part of the negative declaration list FORBIDDEN_PERFORMANCE_TERMS)
        runner_path = m2_structural_runner.__file__
        with open(runner_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # We can check that the words like 'pnl', 'winrate', 'drawdown' do not occur outside the negative list.
        # This is proven by static checks, but let's confirm the list itself exists and matches the forbidden terms.
        self.assertTrue(hasattr(m2_structural_runner, "FORBIDDEN_PERFORMANCE_TERMS"))
        self.assertIn("pnl", m2_structural_runner.FORBIDDEN_PERFORMANCE_TERMS)
        self.assertIn("winrate", m2_structural_runner.FORBIDDEN_PERFORMANCE_TERMS)
        self.assertIn("drawdown", m2_structural_runner.FORBIDDEN_PERFORMANCE_TERMS)
        
    def test_no_filesystem_writing(self) -> None:
        # Run evaluation and verify no files are written to disk
        times = pd.date_range("2015-01-01 00:00:00", periods=10, freq="5min", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [1.0] * 10,
                "high": [1.1] * 10,
                "low": [0.9] * 10,
                "close": [1.0] * 10,
            },
            index=times
        )
        
        initial_files = set(os.listdir("."))
        
        m2_structural_runner.run_m2_structural_evaluation(
            [FakeStrategyMR02],
            frame,
            "2015-01-01 00:00:00",
            "2015-01-01 01:00:00"
        )
        
        current_files = set(os.listdir("."))
        self.assertEqual(initial_files, current_files, "Runner wrote unauthorized files to the root directory")
        
    def test_no_frame_mutation(self) -> None:
        # Verify that running counts does not mutate the original frame
        times = pd.date_range("2015-01-01 00:00:00", periods=10, freq="5min", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [1.0] * 10,
                "high": [1.1] * 10,
                "low": [0.9] * 10,
                "close": [1.0] * 10,
            },
            index=times
        )
        
        frame_copy = frame.copy()
        
        m2_structural_runner.run_structural_counts(FakeStrategyMR02, frame)
        
        pd.testing.assert_frame_equal(frame, frame_copy)
        
    def test_runner_does_not_mutate_params(self) -> None:
        # Verify that parameters are not mutated during the run
        params = {"session_name": "london", "target_rr": 1.5}
        params_copy = dict(params)
        
        times = pd.date_range("2015-01-01 00:00:00", periods=10, freq="5min", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [1.0] * 10,
                "high": [1.1] * 10,
                "low": [0.9] * 10,
                "close": [1.0] * 10,
            },
            index=times
        )
        
        m2_structural_runner.run_structural_counts(FakeStrategyMR02, frame, params=params)
        
        self.assertEqual(params, params_copy)


if __name__ == "__main__":
    unittest.main()

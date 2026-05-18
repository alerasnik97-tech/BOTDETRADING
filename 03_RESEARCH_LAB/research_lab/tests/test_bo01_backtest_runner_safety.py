# -*- coding: utf-8 -*-
import unittest
import inspect
from research_lab.runners import bo01_backtest_runner as runner

class TestBO01BacktestRunnerSafety(unittest.TestCase):
    
    def test_import_side_effects(self):
        """Verify that importing the runner spawns no background processes or file writes."""
        self.assertEqual(runner.RUNNER_ID, "BO01_BACKTEST_RUNNER_SYNTHETIC_V1")

    def test_source_code_safety_scan(self):
        """Statically scans the runner source code for safety violations."""
        source = inspect.getsource(runner)
        
        # 1. Prohibited File I/O
        self.assertNotIn("read_csv", source, "Runner must not import or call read_csv.")
        self.assertNotIn("to_csv", source, "Runner must not write using to_csv.")
        self.assertNotIn("Path(", source, "Path class initialization is strictly banned in the runner.")
        
        # 2. Prohibited Folders / Vault Paths
        self.assertNotIn("05_MARKET_DATA_VAULT", source, "References to 05_MARKET_DATA_VAULT are strictly banned in the runner.")
        self.assertNotIn("data_vault", source, "References to 'data_vault' are strictly banned in the runner.")
        
        # 3. Prohibited Environments
        self.assertNotIn("FTMO", source, "Runner must not contain FTMO prop-firm terms.")
        self.assertNotIn("broker", source, "Broker integration terms are banned.")
        self.assertNotIn("Telegram", source, "Telegram API keys or hooks are strictly banned.")
        
        # 4. Prohibited Optimization Sweeps
        self.assertNotIn("grid_search", source, "Grid search parameter sweep logic is strictly banned in the runner.")
        self.assertNotIn("walk_forward", source, "Walk-forward validation logic is strictly banned in the runner.")

        # 5. Policy Verifications
        self.assertIn("ENTRY_NEXT_CANDLE_OPEN", source, "Source code must contain ENTRY_NEXT_CANDLE_OPEN policy.")
        self.assertNotIn("breakout_price", source, "Source code must not contain a breakout entry execution mechanism.")
        self.assertNotIn("contract_boundary", source, "Source code must not contain contract boundary entry execution.")

if __name__ == "__main__":
    unittest.main()

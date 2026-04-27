import unittest
import os
import sys
import pandas as pd

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from shadow_line_lab.checkpoint_review import evaluator, config

class TestCheckpointReview(unittest.TestCase):
    def test_evaluator_no_data(self):
        ev = evaluator.CheckpointEvaluator()
        res = ev.evaluate()
        self.assertEqual(res["decision"], "CHECKPOINT_NOT_REACHED")

    def test_checkpoint_logic(self):
        ev = evaluator.CheckpointEvaluator()
        # Mocking data for N=5
        mock_ledger = pd.DataFrame([
            {"classification": "TRADE_EXECUTED", "pnl_r": 1.0},
            {"classification": "TRADE_EXECUTED", "pnl_r": 1.0},
            {"classification": "TRADE_EXECUTED", "pnl_r": 1.0},
            {"classification": "TRADE_EXECUTED", "pnl_r": 1.0},
            {"classification": "TRADE_EXECUTED", "pnl_r": 1.0}
        ])
        ev.load_ledger = lambda: mock_ledger
        res = ev.evaluate()
        self.assertEqual(res["checkpoint_target"], 5)
        self.assertEqual(res["decision"], "CONTINUE_INCUBATION")

    def test_hold_logic(self):
        ev = evaluator.CheckpointEvaluator()
        # Mocking data for N=10 with high DD
        mock_ledger = pd.DataFrame([{"classification": "TRADE_EXECUTED", "pnl_r": -1.0}] * 11)
        ev.load_ledger = lambda: mock_ledger
        res = ev.evaluate()
        self.assertEqual(res["decision"], "HOLD_SHADOW")

if __name__ == "__main__":
    unittest.main()

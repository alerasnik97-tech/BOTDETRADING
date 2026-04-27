import unittest
import os
import sys
import pandas as pd
import json

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from shadow_line_lab.evidence_tribunal import evaluator, config, reporting

class TestEvidenceTribunal(unittest.TestCase):
    def test_evaluator_loading(self):
        trib = evaluator.ShadowEvidenceTribunal()
        # Si no hay ledger, debe dar veredicto de error/hold
        scorecard = trib.evaluate()
        self.assertIn(scorecard["verdict"], ["SHADOW_HOLD", "SHADOW_INCUBATING"])

    def test_metric_calculation_logic(self):
        # Crear un ledger temporal para testear cálculos
        test_ledger = pd.DataFrame([
            {"date": "2026-04-20", "classification": "TRADE_EXECUTED", "pnl_r": 1.5, "timeout_flag": False, "news_blocked": False},
            {"date": "2026-04-21", "classification": "TRADE_EXECUTED", "pnl_r": -1.0, "timeout_flag": False, "news_blocked": False},
            {"date": "2026-04-22", "classification": "TRADE_EXECUTED", "pnl_r": 2.0, "timeout_flag": False, "news_blocked": False}
        ])
        temp_ledger_path = os.path.join(config.TRIBUNAL_DIR, "tests", "temp_ledger.csv")
        test_ledger.to_csv(temp_ledger_path, index=False)
        
        # Mocking el ledger file en el evaluator si fuera necesario, pero aquí testeamos el flujo
        trib = evaluator.ShadowEvidenceTribunal()
        # Manual inject metrics or mock load_ledger
        trib.load_ledger = lambda: test_ledger
        scorecard = trib.evaluate()
        
        self.assertEqual(scorecard["metrics"]["total_shadow_trades"], 3)
        self.assertEqual(scorecard["metrics"]["cumulative_R"], 2.5)
        
        os.remove(temp_ledger_path)

    def test_report_generation(self):
        mock_scorecard = {
            "timestamp": "2026-04-24T00:00:00Z",
            "verdict": "SHADOW_INCUBATING",
            "metrics": {"total_shadow_trades": 1, "win_rate": 100},
            "alerts": []
        }
        reporting.generate_reports(mock_scorecard)
        self.assertTrue(os.path.exists(config.SCORECARD_JSON))
        self.assertTrue(os.path.exists(config.SCORECARD_MD))

if __name__ == "__main__":
    unittest.main()

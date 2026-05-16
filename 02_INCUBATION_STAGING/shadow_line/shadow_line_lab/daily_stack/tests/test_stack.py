import unittest
import os
import sys
import pandas as pd

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from shadow_line_lab.daily_stack import run_shadow_daily_stack, config

class TestShadowDailyStack(unittest.TestCase):
    def test_stack_dry_run(self):
        stack = run_shadow_daily_stack.ShadowDailyStack()
        status = stack.run_complete_stack(date_str="2026-04-24")
        
        self.assertEqual(status, "SHADOW_STACK_OK")
        
        # Verificar archivos
        self.assertTrue(os.path.exists(config.OPERATIONAL_LOG))
        self.assertTrue(os.path.exists(config.DAILY_SCORECARD_JSON))
        self.assertTrue(os.path.exists(config.INCUBATION_SUMMARY_MD))

    def test_log_integrity(self):
        if os.path.exists(config.OPERATIONAL_LOG):
            df = pd.read_csv(config.OPERATIONAL_LOG)
            self.assertTrue("tribunal_verdict" in df.columns)
            self.assertTrue("shadow_runner_status" in df.columns)

if __name__ == "__main__":
    unittest.main()

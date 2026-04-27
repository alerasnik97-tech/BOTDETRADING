import unittest
import os
import sys
import json
from datetime import datetime

# Agregar el directorio raíz al path para importar el paquete shadow_line_lab
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shadow_line_lab import config, orchestrator, ledger_io

class TestShadowInfrastructure(unittest.TestCase):
    def test_config_isolation(self):
        self.assertTrue(config.SHADOW_DIR.endswith("shadow_line_lab"))
        self.assertIn("tp_1p50_timeout_4h", config.STRATEGY_CONFIG["variant_id"])

    def test_orchestrator_dry_run(self):
        orch = orchestrator.ShadowOrchestrator()
        test_date = "2026-04-23"
        res = orch.run_daily_process(test_date)
        
        self.assertEqual(res["date"], test_date)
        self.assertEqual(res["classification"], "DRY_RUN_NO_DATA")
        
        # Verificar que se crearon los archivos de resultados
        self.assertTrue(os.path.exists(config.LEDGER_FILE))
        self.assertTrue(os.path.exists(config.DAILY_STATUS_FILE))
        self.assertTrue(os.path.exists(config.REPORT_FILE))

    def test_ledger_append(self):
        test_entry = {"date": "2026-04-24", "line_name": "TEST", "classification": "TEST_ENTRY"}
        ledger_io.write_ledger_entry(test_entry, config.LEDGER_FILE)
        
        with open(config.LEDGER_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertTrue(len(lines) >= 2) # Header + at least one entry

    def test_core_untouched(self):
        # Verificar que no existe rastro de la shadow line en el core results principal
        official_results_dir = os.path.join(config.BASE_DIR, "results")
        shadow_file_in_official = os.path.join(official_results_dir, "shadow_ledger.csv")
        self.assertFalse(os.path.exists(shadow_file_in_official))

if __name__ == "__main__":
    unittest.main()

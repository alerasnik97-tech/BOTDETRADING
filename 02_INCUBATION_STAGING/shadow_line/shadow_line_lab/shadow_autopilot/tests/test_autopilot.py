import unittest
import os
import sys

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from shadow_line_lab.shadow_autopilot import coordinator, config

class TestShadowAutopilot(unittest.TestCase):
    def test_pipeline_execution(self):
        coord = coordinator.ShadowCoordinator()
        state = coord.execute_full_pipeline(date_str="2026-04-24")
        
        self.assertEqual(state["overall_status"], "SHADOW_AUTOPILOT_OK")
        self.assertTrue("target_date" in state)
        self.assertTrue("checkpoint_status" in state)

    def test_log_creation(self):
        # El log debe existir tras correr el pipeline (o ya existir de antes)
        self.assertTrue(os.path.exists(config.AUTOPILOT_LOG_CSV))

if __name__ == "__main__":
    unittest.main()

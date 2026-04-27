import unittest
import os
import sys

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from micro_pilot_gate import evaluator

class TestMicroPilotGate(unittest.TestCase):
    def test_evaluator_init(self):
        gate = evaluator.MicroPilotGate()
        res = gate.evaluate()
        self.assertIn(res["verdict"], ["NOT_READY_FOR_MICRO_PILOT", "MICRO_PILOT_ALLOWED"])

    def test_gates_structure(self):
        gate = evaluator.MicroPilotGate()
        res = gate.evaluate()
        self.assertTrue("research_robustness" in res["gates"])
        self.assertTrue("shadow_evidence_min" in res["gates"])

    def test_risk_protocol_presence(self):
        gate = evaluator.MicroPilotGate()
        res = gate.evaluate()
        self.assertTrue("risk_per_trade" in res["risk_protocol"])

if __name__ == "__main__":
    unittest.main()

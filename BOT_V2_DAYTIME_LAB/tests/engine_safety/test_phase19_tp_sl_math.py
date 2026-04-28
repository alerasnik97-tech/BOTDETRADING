import json
import unittest
from pathlib import Path


ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
REPORT = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase19_forensic_audit" / "tp_sl_math" / "phase19_tp_sl_math_audit.json"


class TestPhase19TpSlMath(unittest.TestCase):
    def test_phase19_tp_sl_orientation_and_risk(self):
        data = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(data["long_tp_above_entry_violations"], 0)
        self.assertEqual(data["long_sl_below_entry_violations"], 0)
        self.assertEqual(data["short_tp_below_entry_violations"], 0)
        self.assertEqual(data["short_sl_above_entry_violations"], 0)
        self.assertEqual(data["tp_2_5r_violations"], 0)
        self.assertEqual(data["risk_le_zero"], 0)


if __name__ == "__main__":
    unittest.main()

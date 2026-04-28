import json
import unittest
from pathlib import Path


ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
REPORT = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase19_forensic_audit" / "signal_no_lookahead" / "phase19_signal_no_lookahead_report.json"


class TestPhase19SignalNoLookahead(unittest.TestCase):
    def test_phase19_entry_uses_next_bar_open_and_native_m3(self):
        data = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertTrue(data["entry_occurs_next_bar_open"], "Phase19 must enter on next bar open, not CHOCH close")
        self.assertEqual(data["entry_next_bar_violations"], 0)
        self.assertTrue(data["m3_source_valid_native_m3"], "Phase19 needs certified native M3 or lower-granularity reconstruction")


if __name__ == "__main__":
    unittest.main()

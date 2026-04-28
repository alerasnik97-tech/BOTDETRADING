import json
import unittest
from pathlib import Path


ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
REPORT = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase19_forensic_audit" / "multi_trade" / "phase19_multi_trade_report.json"


class TestPhase19MultiTradeDailyLimit(unittest.TestCase):
    def test_phase19_has_no_duplicate_or_overlapping_trades(self):
        data = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertLessEqual(data["max_daily_trades"], 3)
        self.assertEqual(data["duplicate_same_event_count"], 0)
        self.assertEqual(data["same_timestamp_trade_count"], 0)
        self.assertEqual(data["overlapping_position_count"], 0)
        self.assertEqual(data["trade_not_after_previous_count"], 0)


if __name__ == "__main__":
    unittest.main()

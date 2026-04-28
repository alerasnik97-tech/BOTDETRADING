import json
import unittest
from pathlib import Path


ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
REPORT = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase19_forensic_audit" / "time_news" / "phase19_time_news_report.json"


class TestPhase19TimeNewsGuard(unittest.TestCase):
    def test_phase19_time_news_and_forced_close(self):
        data = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(data["trades_before_0700"], 0)
        self.assertEqual(data["trades_after_2000"], 0)
        self.assertEqual(data["trades_open_after_2000"], 0)
        self.assertTrue(data["news_guard_active_in_legacy_code"])
        self.assertEqual(data["news_violations_30m"], 0)
        self.assertTrue(data["forced_close_correct"])


if __name__ == "__main__":
    unittest.main()

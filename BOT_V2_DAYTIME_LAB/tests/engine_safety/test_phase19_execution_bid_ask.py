import json
import unittest
from pathlib import Path


ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
REPORT = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase19_forensic_audit" / "execution" / "phase19_bid_ask_audit.json"


class TestPhase19ExecutionBidAsk(unittest.TestCase):
    def test_phase19_uses_realistic_bid_ask_and_same_bar_policy(self):
        data = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertTrue(data["long_entry_ask_used"])
        self.assertTrue(data["long_exit_bid_used"])
        self.assertTrue(data["short_entry_bid_used"])
        self.assertTrue(data["short_exit_ask_used"])
        self.assertTrue(data["historical_spread_applied_to_legacy_result"])
        self.assertTrue(data["slippage_applied_to_legacy_result"])
        self.assertTrue(data["same_bar_conservative_policy_implemented"])
        self.assertEqual(data["trades_without_sl"], 0)
        self.assertEqual(data["trades_without_tp"], 0)


if __name__ == "__main__":
    unittest.main()

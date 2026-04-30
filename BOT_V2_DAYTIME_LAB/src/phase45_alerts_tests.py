import unittest
import os
import json
from phase45_alert_engine import AlertEngine
from phase45_telegram_sender import TelegramSender
from phase45_alert_state import AlertState

class TestAlerts(unittest.TestCase):
    def setUp(self):
        self.root = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
        
    def test_config_no_token(self):
        sender = TelegramSender(token="", chat_id="")
        res = sender.send_message("test")
        self.assertEqual(res["status"], "TELEGRAM_NOT_CONFIGURED")

    def test_alert_engine_read_only(self):
        engine = AlertEngine(self.root)
        # Should not crash and should return list
        alerts = engine.detect_alerts()
        self.assertIsInstance(alerts, list)

    def test_dedup_logic(self):
        state_file = os.path.join(self.root, "scratch", "test_alert_state.json")
        if os.path.exists(state_file): os.remove(state_file)
        
        state = AlertState(state_file)
        # First time should send
        self.assertTrue(state.should_send("test_key", "INFO", cooldown_minutes=10))
        # Second time immediate should NOT send
        self.assertFalse(state.should_send("test_key", "INFO", cooldown_minutes=10))
        
        # Cleanup
        if os.path.exists(state_file): os.remove(state_file)

    def test_critical_bypass(self):
        state_file = os.path.join(self.root, "scratch", "test_alert_state_crit.json")
        if os.path.exists(state_file): os.remove(state_file)
        
        state = AlertState(state_file)
        # Critical bypass usually allows more frequent alerts if needed, 
        # but here we test if it follows its own rule.
        self.assertTrue(state.should_send("crit_key", "CRITICAL", cooldown_minutes=10))
        
        # Cleanup
        if os.path.exists(state_file): os.remove(state_file)

if __name__ == "__main__":
    unittest.main()

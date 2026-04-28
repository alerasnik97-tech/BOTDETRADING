
import unittest
import pandas as pd
from datetime import datetime, timezone, timedelta
from news_fortress.news_fortress_gate import NewsFortressGate

class TestNewsFortressGate(unittest.TestCase):
    def setUp(self):
        # Create a mock calendar
        self.mock_data = [
            {
                "timestamp_utc": "2025-04-03T13:30:00+00:00",
                "event": "Non-Farm Employment Change",
                "currency": "USD",
                "impact_level": "HIGH"
            },
            {
                "timestamp_utc": "2025-04-03T15:00:00+00:00",
                "event": "ISM Services PMI",
                "currency": "USD",
                "impact_level": "HIGH"
            },
            {
                "timestamp_utc": "2025-04-04T12:30:00+00:00",
                "event": "NFP",
                "currency": "USD",
                "impact_level": "HIGH"
            }
        ]
        self.df = pd.DataFrame(self.mock_data)
        self.df['timestamp_utc'] = pd.to_datetime(self.df['timestamp_utc'], utc=True)
        self.gate = NewsFortressGate(self.df)

    def test_block_high_impact(self):
        # 13:30 is high impact. Test at 13:00 (within 60m)
        now = datetime(2025, 4, 3, 13, 0, tzinfo=timezone.utc)
        allow, reason = self.gate.evaluate_trading_permission(now)
        self.assertFalse(allow)
        self.assertIn("BLOCK", reason)

    def test_allow_safe_time(self):
        # Test at 11:00 (more than 120m from 13:30)
        now = datetime(2025, 4, 3, 11, 0, tzinfo=timezone.utc)
        allow, reason = self.gate.evaluate_trading_permission(now)
        self.assertTrue(allow)
        self.assertIn("ALLOW", reason)

    def test_block_ultra_critical(self):
        # NFP is at 12:30. Test at 11:00 (within 120m ultra buffer)
        now = datetime(2025, 4, 4, 11, 0, tzinfo=timezone.utc)
        allow, reason = self.gate.evaluate_trading_permission(now)
        self.assertFalse(allow)
        self.assertIn("BLOCK", reason)
        self.assertIn("120m", reason)

    def test_fail_closed_empty_feed(self):
        empty_gate = NewsFortressGate(pd.DataFrame())
        now = datetime(2025, 4, 3, 13, 0, tzinfo=timezone.utc)
        allow, reason = empty_gate.evaluate_trading_permission(now)
        self.assertFalse(allow)
        self.assertIn("UNHEALTHY", reason)

if __name__ == "__main__":
    unittest.main()

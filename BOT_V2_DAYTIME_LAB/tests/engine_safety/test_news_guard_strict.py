from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from news_guard_strict import NewsGuardStrict, audit_news_feed


class NewsGuardStrictTests(unittest.TestCase):
    def test_high_impact_usd_blocks_inside_buffer(self):
        guard = NewsGuardStrict(
            pd.DataFrame(
                [
                    {
                        "timestamp_utc": "2026-01-02T13:30:00+00:00",
                        "currency": "USD",
                        "impact_level": "HIGH",
                        "event_name_normalized": "non-farm employment change",
                    }
                ]
            )
        )
        decision = guard.should_block(pd.Timestamp("2026-01-02T13:45:00+00:00"), 30)
        self.assertTrue(decision.blocked)

    def test_high_impact_eur_blocks_inside_buffer(self):
        guard = NewsGuardStrict(
            pd.DataFrame(
                [
                    {
                        "timestamp_utc": "2026-03-12T13:15:00+00:00",
                        "currency": "EUR",
                        "impact_level": "HIGH",
                        "event_name_normalized": "ecb rate decision",
                    }
                ]
            )
        )
        self.assertTrue(guard.should_block(pd.Timestamp("2026-03-12T13:45:00+00:00"), 45).blocked)

    def test_unknown_event_blocks(self):
        guard = NewsGuardStrict(
            pd.DataFrame(
                [
                    {
                        "timestamp_utc": "2026-01-02T13:30:00+00:00",
                        "currency": "USD",
                        "impact_level": "UNKNOWN",
                        "event_name_normalized": "ambiguous",
                    }
                ]
            )
        )
        self.assertTrue(guard.should_block(pd.Timestamp("2026-01-02T14:30:00+00:00"), 30).blocked)

    def test_invalid_timestamp_blocks(self):
        guard = NewsGuardStrict(
            pd.DataFrame(
                [
                    {
                        "timestamp_utc": "not-a-date",
                        "currency": "USD",
                        "impact_level": "HIGH",
                        "event_name_normalized": "cpi",
                    }
                ]
            )
        )
        self.assertTrue(guard.should_block(pd.Timestamp("2026-01-02T14:30:00+00:00"), 30).blocked)

    def test_missing_feed_invalidates_audit(self):
        summary, _, _ = audit_news_feed(Path(tempfile.gettempdir()) / "missing_news_feed_for_test.csv")
        self.assertEqual(summary["verdict"], "NEWS_GUARD_INVALIDATED")

    def test_low_impact_outside_usd_eur_ignored(self):
        guard = NewsGuardStrict(
            pd.DataFrame(
                [
                    {
                        "timestamp_utc": "2026-01-02T13:30:00+00:00",
                        "currency": "JPY",
                        "impact_level": "LOW",
                        "event_name_normalized": "minor event",
                    }
                ]
            )
        )
        self.assertFalse(guard.should_block(pd.Timestamp("2026-01-02T13:30:00+00:00"), 60).blocked)

    def test_dst_and_buffers_work(self):
        guard = NewsGuardStrict(
            pd.DataFrame(
                [
                    {
                        "timestamp_utc": "2026-03-08T13:30:00+00:00",
                        "currency": "USD",
                        "impact_level": "HIGH",
                        "event_name_normalized": "core cpi",
                    }
                ]
            )
        )
        self.assertFalse(guard.should_block(pd.Timestamp("2026-03-08T12:29:00+00:00"), 60).blocked)
        self.assertTrue(guard.should_block(pd.Timestamp("2026-03-08T12:45:00+00:00"), 45).blocked)
        self.assertTrue(guard.should_block(pd.Timestamp("2026-03-08T13:00:00+00:00"), 30).blocked)

    def test_duplicate_events_deduplicate(self):
        rows = [
            {
                "timestamp_utc": "2026-01-02T13:30:00+00:00",
                "currency": "USD",
                "impact_level": "HIGH",
                "event_name_normalized": "cpi",
            }
        ]
        guard = NewsGuardStrict(pd.DataFrame(rows + rows))
        self.assertEqual(guard.should_block(pd.Timestamp("2026-01-02T13:30:00+00:00"), 30).matched_events, 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from data_certification.m3_gap_repair import classify_gap
from news_guard_strict import normalize_news


class NewsGapInteractionTests(unittest.TestCase):
    def test_news_high_impact_gap_bloquea(self):
        news = normalize_news(
            pd.DataFrame(
                [
                    {
                        "timestamp_utc": "2026-01-06T15:00:00+00:00",
                        "currency": "USD",
                        "impact_level": "HIGH",
                        "event_name_normalized": "cpi",
                    }
                ]
            )
        )
        row = pd.Series(
            {
                "start_utc": pd.Timestamp("2026-01-06T15:03:00+00:00"),
                "end_utc": pd.Timestamp("2026-01-06T15:06:00+00:00"),
                "bid_missing": True,
                "ask_missing": True,
                "both_missing": True,
            }
        )
        result = classify_gap(row, news)
        self.assertTrue(result["near_high_impact_news_30m"])
        self.assertEqual(result["classification"], "NEWS_WINDOW_GAP")
        self.assertEqual(result["severity"], "CRITICAL_MASK_DAY")


if __name__ == "__main__":
    unittest.main()

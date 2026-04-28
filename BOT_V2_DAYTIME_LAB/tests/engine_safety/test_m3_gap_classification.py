from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from data_certification.m3_gap_repair import classify_gap


def row(start_utc: str, end_utc: str) -> pd.Series:
    return pd.Series(
        {
            "start_utc": pd.Timestamp(start_utc),
            "end_utc": pd.Timestamp(end_utc),
            "bid_missing": True,
            "ask_missing": True,
            "both_missing": True,
        }
    )


class M3GapClassificationTests(unittest.TestCase):
    def test_weekend_gap_no_bloquea(self):
        result = classify_gap(row("2026-01-03T12:00:00+00:00", "2026-01-03T12:06:00+00:00"))
        self.assertEqual(result["classification"], "WEEKEND_MARKET_CLOSED")
        self.assertEqual(result["severity"], "IGNORE_SAFE")

    def test_outside_window_gap_no_bloquea_phase19(self):
        result = classify_gap(row("2026-01-06T03:00:00+00:00", "2026-01-06T03:06:00+00:00"))
        self.assertFalse(result["in_phase19_window_08_1630"])
        self.assertEqual(result["classification"], "OUTSIDE_USER_WINDOW")

    def test_in_window_critical_gap_bloquea(self):
        result = classify_gap(row("2026-01-06T15:00:00+00:00", "2026-01-06T15:06:00+00:00"))
        self.assertTrue(result["in_phase19_window_08_1630"])
        self.assertEqual(result["severity"], "CRITICAL_MASK_DAY")

    def test_unknown_gap_fails_closed(self):
        result = classify_gap(row("2026-01-06T23:30:00+00:00", "2026-01-06T23:36:00+00:00"))
        self.assertIn(result["severity"], {"WARNING_MASK_SESSION", "CRITICAL_BLOCK_CERTIFICATION"})


if __name__ == "__main__":
    unittest.main()

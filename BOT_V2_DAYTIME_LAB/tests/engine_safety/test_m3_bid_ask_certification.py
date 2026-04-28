from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from data_certification.m3_bid_ask_builder import audit_m1_source, gap_report, reject_source_for_m3


def frame(start="2026-01-05 12:00:00+00:00", periods=5, ask_offset=0.0001):
    ts = pd.date_range(start, periods=periods, freq="min", tz="UTC")
    bid = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [1.1000] * periods,
            "high": [1.1002] * periods,
            "low": [1.0998] * periods,
            "close": [1.1001] * periods,
            "volume": [1] * periods,
        }
    )
    ask = bid.copy()
    for col in ["open", "high", "low", "close"]:
        ask[col] = ask[col] + ask_offset
    return bid, ask


class M3BidAskCertificationTests(unittest.TestCase):
    def test_no_valid_source_blocks_m5(self):
        self.assertEqual(reject_source_for_m3("m5", True, True), "SOURCE_REJECTED")

    def test_m3_without_ask_blocks(self):
        self.assertEqual(reject_source_for_m3("m1", True, False), "SOURCE_MISSING")

    def test_bid_above_ask_fails(self):
        bid, ask = frame(ask_offset=-0.0001)
        self.assertEqual(audit_m1_source(bid, ask)["verdict"], "SOURCE_REQUIRES_REPAIR")

    def test_duplicate_timestamp_fails(self):
        bid, ask = frame()
        bid.loc[1, "timestamp"] = bid.loc[0, "timestamp"]
        self.assertEqual(audit_m1_source(bid, ask)["verdict"], "SOURCE_REQUIRES_REPAIR")

    def test_ohlc_invalid_fails(self):
        bid, ask = frame()
        bid.loc[0, "high"] = bid.loc[0, "low"] - 0.0001
        self.assertEqual(audit_m1_source(bid, ask)["verdict"], "SOURCE_REQUIRES_REPAIR")

    def test_0700_2000_ny_continuous_coverage_has_no_gap(self):
        ts = pd.date_range("2026-01-05 12:00:00+00:00", periods=781, freq="min", tz="UTC")
        self.assertTrue(gap_report(pd.Series(ts), 1).empty)


if __name__ == "__main__":
    unittest.main()

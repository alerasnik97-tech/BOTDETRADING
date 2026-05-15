import os
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline

P = load_pipeline()


class TestDateGuardRobust(unittest.TestCase):
    def test_date_guard_detects_iso_2025(self):
        ok, errs = P.check_temporal_no_2025_2026(
            ["signal_time"], [{"signal_time": "2025-01-01"}])
        self.assertFalse(ok)
        self.assertTrue(any("2025" in e for e in errs))

    def test_date_guard_detects_compact_20250101(self):
        ok, errs = P.check_temporal_no_2025_2026(
            ["signal_time"], [{"signal_time": "20250101"}])
        self.assertFalse(ok)
        self.assertTrue(any("20250101" in e for e in errs))

    def test_date_guard_detects_epoch_seconds_2025(self):
        ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
        ok, errs = P.check_temporal_no_2025_2026(
            ["timestamp"], [{"timestamp": str(ts)}])
        self.assertFalse(ok)
        self.assertTrue(any("year 2025" in e for e in errs))

    def test_date_guard_detects_epoch_ms_2026(self):
        ts = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1000
        ok, errs = P.check_temporal_no_2025_2026(
            ["timestamp_ms"], [{"timestamp_ms": str(ts)}])
        self.assertFalse(ok)
        self.assertTrue(any("year 2026" in e for e in errs))

    def test_date_guard_ignores_run_id_with_2025(self):
        ok, errs = P.check_temporal_no_2025_2026(
            ["run_id"], [{"run_id": "RB2025abcdef"}])
        self.assertTrue(ok, errs)

    def test_date_guard_ignores_non_time_numeric_2025(self):
        ok, errs = P.check_temporal_no_2025_2026(
            ["score"], [{"score": "2025"}])
        self.assertTrue(ok, errs)


if __name__ == "__main__":
    unittest.main()

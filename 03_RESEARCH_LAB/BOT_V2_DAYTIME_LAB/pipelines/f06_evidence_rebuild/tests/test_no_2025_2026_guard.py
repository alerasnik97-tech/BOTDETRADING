import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline, fx

P = load_pipeline()


class TestNo20252026Guard(unittest.TestCase):
    def test_clean_ledger_passes(self):
        header, rows = P.read_csv(fx("synthetic_clean_ledger.csv"))
        ok, errs = P.check_ledger_no_2025_2026(header, rows)
        self.assertTrue(ok, errs)

    def test_2025_rows_fail(self):
        header, rows = P.read_csv(fx("synthetic_bad_2025_rows.csv"))
        ok, errs = P.check_ledger_no_2025_2026(header, rows)
        self.assertFalse(ok)

    def test_scalar_values(self):
        self.assertTrue(P.check_no_2025_2026(["2020-03", "2024-04"])[0])
        self.assertFalse(P.check_no_2025_2026(["2025-01-01"])[0])
        self.assertFalse(P.check_no_2025_2026(["x2026y"])[0])


if __name__ == "__main__":
    unittest.main()

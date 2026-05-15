import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline, fx

P = load_pipeline()


class TestSampleSizeFloor(unittest.TestCase):
    def test_n10_fails(self):
        ok, errs = P.check_sample_size_floor(10, 100)
        self.assertFalse(ok)
        self.assertTrue(any("below institutional floor" in e for e in errs))

    def test_n100_passes(self):
        self.assertTrue(P.check_sample_size_floor(100, 100)[0])

    def test_clean_ledger_meets_floor(self):
        _, rows = P.read_csv(fx("synthetic_clean_ledger.csv"))
        n = sum(1 for r in rows if r.get("family_id") == "F06")
        self.assertGreaterEqual(n, 100)
        self.assertTrue(P.check_sample_size_floor(n, 100)[0])

    def test_non_int_fails_closed(self):
        ok, _ = P.check_sample_size_floor("abc", 100)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()

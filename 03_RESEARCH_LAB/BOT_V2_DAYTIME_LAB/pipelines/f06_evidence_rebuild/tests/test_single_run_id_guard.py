import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline, fx

P = load_pipeline()


class TestSingleRunIdGuard(unittest.TestCase):
    def test_clean_ledger_single_run_id_passes(self):
        _, rows = P.read_csv(fx("synthetic_clean_ledger.csv"))
        ok, errs = P.check_single_run_id(rows)
        self.assertTrue(ok, errs)

    def test_multi_run_id_ledger_fails(self):
        _, rows = P.read_csv(fx("synthetic_bad_multi_runid_ledger.csv"))
        ok, errs = P.check_single_run_id(rows)
        self.assertFalse(ok)
        self.assertTrue(any("multiple run_ids" in e for e in errs))

    def test_empty_rows_fail_closed(self):
        ok, errs = P.check_single_run_id([])
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()

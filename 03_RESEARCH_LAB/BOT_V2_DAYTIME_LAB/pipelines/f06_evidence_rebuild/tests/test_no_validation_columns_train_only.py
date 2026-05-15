import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline, fx

P = load_pipeline()


class TestNoValidationColumns(unittest.TestCase):
    def test_clean_header_passes(self):
        header, _ = P.read_csv(fx("synthetic_clean_ledger.csv"))
        ok, errs = P.check_no_validation_columns(header)
        self.assertTrue(ok, errs)

    def test_validation_columns_fail(self):
        header, _ = P.read_csv(fx("synthetic_bad_validation_columns.csv"))
        ok, errs = P.check_no_validation_columns(header)
        self.assertFalse(ok)

    def test_each_forbidden_column_detected(self):
        for c in ["N_val", "PF_val", "Total_R_val", "WR_val",
                  "val_pass", "combined_pass"]:
            ok, _ = P.check_no_validation_columns(["family_id", c])
            self.assertFalse(ok, f"{c} must be rejected in train-only")


if __name__ == "__main__":
    unittest.main()

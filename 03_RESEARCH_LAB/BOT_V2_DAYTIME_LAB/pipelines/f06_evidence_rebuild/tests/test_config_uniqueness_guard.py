import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline, fx

P = load_pipeline()
RESULT_COLS = ["N_train", "PF_train", "Total_R_train", "WR_train"]


class TestConfigUniquenessGuard(unittest.TestCase):
    def test_degenerate_fixture_fails(self):
        _, rows = P.read_csv(fx("synthetic_bad_duplicate_configs.csv"))
        ok, errs = P.check_config_uniqueness(rows, RESULT_COLS)
        self.assertFalse(ok)
        self.assertTrue(any("degenerate" in e for e in errs))

    def test_varied_configs_pass(self):
        rows = [{"config_id": f"F06_{i:04d}", "N_train": 100 + i,
                 "PF_train": 1.5 + i * 0.01, "Total_R_train": 10 + i,
                 "WR_train": 0.5} for i in range(20)]
        ok, errs = P.check_config_uniqueness(rows, RESULT_COLS)
        self.assertTrue(ok, errs)

    def test_explicit_deduplicated_flag_passes(self):
        rows = [{"config_id": f"F06_{i:04d}", "N_train": 125, "PF_train": 2.2,
                 "Total_R_train": 6.0, "WR_train": 0.5, "deduplicated": "true"}
                for i in range(50)]
        ok, errs = P.check_config_uniqueness(rows, RESULT_COLS)
        self.assertTrue(ok, errs)


if __name__ == "__main__":
    unittest.main()

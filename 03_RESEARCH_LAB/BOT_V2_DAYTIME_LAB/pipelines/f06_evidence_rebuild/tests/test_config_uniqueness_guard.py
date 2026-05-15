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

    def test_config_uniqueness_rejects_duplicate_parameter_hashes(self):
        rows = [{"family_id": "F06", "config_id": f"F06_{i:04d}",
                 "parameter_hash": "same-parameter-hash",
                 "result_signature": f"sig-{i:04d}",
                 "N_train": 120 + i, "PF_train": 1.2 + i * 0.01,
                 "Total_R_train": 5 + i, "WR_train": 0.45 + i * 0.001}
                for i in range(10)]
        ok, errs = P.check_config_uniqueness(rows, RESULT_COLS)
        self.assertFalse(ok)
        self.assertTrue(any("parameter_hash" in e for e in errs))

    def test_config_uniqueness_rejects_duplicate_result_signatures(self):
        rows = [{"family_id": "F06", "config_id": f"F06_{i:04d}",
                 "parameter_hash": f"ph-{i:04d}", "result_signature": "same-result",
                 "N_train": 120 + i, "PF_train": 1.5,
                 "Total_R_train": 9.0, "WR_train": 0.5}
                for i in range(10)]
        ok, errs = P.check_config_uniqueness(rows, RESULT_COLS)
        self.assertFalse(ok)
        self.assertTrue(any("result" in e for e in errs))

    def test_config_uniqueness_allows_explicit_deduplicated_true(self):
        rows = [{"family_id": "F06", "config_id": f"F06_{i:04d}",
                 "parameter_hash": "same-parameter-hash",
                 "result_signature": "same-result",
                 "N_train": 130, "PF_train": 1.3,
                 "Total_R_train": 7.0, "WR_train": 0.48,
                 "deduplicated": "true"}
                for i in range(10)]
        ok, errs = P.check_config_uniqueness(rows, RESULT_COLS)
        self.assertTrue(ok, errs)

    def test_config_uniqueness_accepts_diverse_configs(self):
        rows = [{"family_id": "F06", "config_id": f"F06_{i:04d}",
                 "parameter_hash": f"ph-{i:04d}",
                 "result_signature": f"sig-{i:04d}",
                 "N_train": 140 + i, "PF_train": 1.1 + i * 0.02,
                 "Total_R_train": 3.0 + i, "WR_train": 0.43 + i * 0.002}
                for i in range(10)]
        ok, errs = P.check_config_uniqueness(rows, RESULT_COLS)
        self.assertTrue(ok, errs)


if __name__ == "__main__":
    unittest.main()

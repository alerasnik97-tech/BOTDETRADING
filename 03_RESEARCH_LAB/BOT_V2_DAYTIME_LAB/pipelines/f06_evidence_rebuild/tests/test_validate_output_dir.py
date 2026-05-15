import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline, fx

P = load_pipeline()


def validate_fixture(name):
    output_dir = fx(name)
    return P.validate_output_dir(output_dir, None, None)


class TestValidateOutputDir(unittest.TestCase):
    def test_validate_output_dir_good_passes(self):
        res = validate_fixture("output_good")
        self.assertTrue(res["ok"], res["errors"])
        self.assertEqual(res["decision"], "READY_FOR_CLAUDE_AUDIT")

    def test_validate_output_dir_rejects_multi_runid(self):
        res = validate_fixture("output_bad_multi_runid")
        self.assertFalse(res["ok"])
        self.assertTrue(any("multiple run_ids" in e for e in res["errors"]))

    def test_validate_output_dir_rejects_validation_columns(self):
        res = validate_fixture("output_bad_validation_columns")
        self.assertFalse(res["ok"])
        self.assertTrue(any("validation column" in e for e in res["errors"]))

    def test_validate_output_dir_rejects_2025(self):
        res = validate_fixture("output_bad_2025")
        self.assertFalse(res["ok"])
        self.assertTrue(any("2025" in e for e in res["errors"]))

    def test_validate_output_dir_rejects_hash_mismatch(self):
        res = validate_fixture("output_bad_hash_mismatch")
        self.assertFalse(res["ok"])
        self.assertTrue(any("mismatch" in e for e in res["errors"]))

    def test_validate_output_dir_rejects_sample_size_below_floor(self):
        res = validate_fixture("output_bad_sample_size")
        self.assertFalse(res["ok"])
        self.assertTrue(any("sample size" in e for e in res["errors"]))

    def test_validate_output_dir_rejects_cost_missing_spread(self):
        res = validate_fixture("output_bad_cost_missing_spread")
        self.assertFalse(res["ok"])
        self.assertTrue(any("spread_component" in e for e in res["errors"]))

    def test_validate_output_dir_rejects_quarantined_path(self):
        res = validate_fixture("output_bad_quarantined_manifest_path")
        self.assertFalse(res["ok"])
        self.assertTrue(any("quarantined token" in e for e in res["errors"]))


if __name__ == "__main__":
    unittest.main()

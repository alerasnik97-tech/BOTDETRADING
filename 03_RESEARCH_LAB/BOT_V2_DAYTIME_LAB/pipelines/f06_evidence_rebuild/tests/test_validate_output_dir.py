import json
import os
import shutil
import sys
import tempfile
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

    def test_fixture_current_script_sha_marker_allowed_only_in_fixtures(self):
        manifest_path = os.path.join(fx("output_good"), "MANIFEST_good.json")
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        self.assertEqual(manifest["script_sha256"], P.CURRENT_SCRIPT_SHA256_MARKER)
        res = validate_fixture("output_good")
        self.assertTrue(res["ok"], res["errors"])

    def test_current_script_sha_marker_forbidden_outside_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied = os.path.join(tmp, "real_output_with_marker")
            shutil.copytree(fx("output_good"), copied)
            res = P.validate_output_dir(copied, None, None)
        self.assertFalse(res["ok"])
        self.assertTrue(any("marker is forbidden outside" in e for e in res["errors"]), res["errors"])

    def test_output_good_passes_after_script_change_without_manual_hash_edit(self):
        manifest_path = os.path.join(fx("output_good"), "MANIFEST_good.json")
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        self.assertEqual(manifest["script_sha256"], P.CURRENT_SCRIPT_SHA256_MARKER)
        self.assertNotEqual(manifest["script_sha256"], P.sha256_file(P.__file__))
        res = validate_fixture("output_good")
        self.assertTrue(res["ok"], res["errors"])

    def test_real_manifest_requires_physical_script_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied = os.path.join(tmp, "real_output_with_physical_hash")
            shutil.copytree(fx("output_good"), copied)
            manifest_path = os.path.join(copied, "MANIFEST_good.json")
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            manifest["script_sha256"] = P.sha256_file(P.__file__)
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2)
            res = P.validate_output_dir(copied, None, None)
        self.assertTrue(res["ok"], res["errors"])

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

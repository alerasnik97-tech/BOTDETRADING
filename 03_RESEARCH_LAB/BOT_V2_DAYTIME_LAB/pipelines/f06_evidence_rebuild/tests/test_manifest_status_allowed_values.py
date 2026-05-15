import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline

P = load_pipeline()


def base():
    from test_manifest_schema import valid_manifest
    return valid_manifest()


class TestManifestStatusAllowedValues(unittest.TestCase):
    def test_allowed_status_values_accepted(self):
        for s in ("DRY_RUN_SCHEMA_VALIDATED", "BLOCKED_GUARD_FAILED",
                  "READY_FOR_CLEAN_TRAIN_RERUN", "NOT_READY"):
            m = base()
            m["status"] = s
            ok, errs = P.validate_manifest(m)
            self.assertTrue(ok, f"{s} should be accepted: {errs}")

    def test_arbitrary_status_rejected(self):
        for s in ("CERTIFIED", "READY_FOR_VAL", "COST_ROBUST", "", "OK", "PASS"):
            m = base()
            m["status"] = s
            ok, errs = P.validate_manifest(m)
            self.assertFalse(ok, f"status {s!r} must be rejected")


if __name__ == "__main__":
    unittest.main()

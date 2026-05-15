import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline

P = load_pipeline()


def base():
    from test_manifest_schema import valid_manifest
    return valid_manifest()


class TestManifestHashHexEnforced(unittest.TestCase):
    """Closes the fake-hash gap: a non-sha256 string in output_hashes /
    script_sha256 / config_sha256 must be rejected (the V50B manifest was
    a 4-line stub with no real hashes)."""

    def test_fake_output_hash_rejected(self):
        m = base()
        m["output_hashes"] = {"ledger.csv": "not-a-real-hash"}
        ok, errs = P.validate_manifest(m)
        self.assertFalse(ok)
        self.assertTrue(any("not a sha256" in e for e in errs))

    def test_short_hash_rejected(self):
        m = base()
        m["output_hashes"] = {"ledger.csv": "abc123"}
        ok, _ = P.validate_manifest(m)
        self.assertFalse(ok)

    def test_bad_script_sha_rejected(self):
        m = base()
        m["script_sha256"] = "deadbeef"
        ok, _ = P.validate_manifest(m)
        self.assertFalse(ok)

    def test_valid_64hex_accepted(self):
        m = base()
        m["output_hashes"] = {"ledger.csv": "f" * 64}
        m["script_sha256"] = "a" * 64
        m["config_sha256"] = "b" * 64
        ok, errs = P.validate_manifest(m)
        self.assertTrue(ok, errs)


if __name__ == "__main__":
    unittest.main()

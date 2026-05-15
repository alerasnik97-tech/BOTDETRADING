import hashlib
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline, fx

P = load_pipeline()


class TestManifestHashes(unittest.TestCase):
    def test_sha256_file_matches_hashlib(self):
        path = fx("synthetic_cost_model_sample.csv")
        with open(path, "rb") as fh:
            expected = hashlib.sha256(fh.read()).hexdigest()
        self.assertEqual(P.sha256_file(path), expected)

    def test_sha256_bytes_deterministic(self):
        self.assertEqual(P.sha256_bytes(b"abc"), P.sha256_bytes(b"abc"))
        self.assertNotEqual(P.sha256_bytes(b"abc"), P.sha256_bytes(b"abd"))

    def test_manifest_output_hash_roundtrip(self):
        path = fx("synthetic_clean_ledger.csv")
        digest = P.sha256_file(path)
        manifest_hashes = {"synthetic_clean_ledger.csv": digest}
        # tamper detection
        recomputed = P.sha256_file(path)
        self.assertEqual(manifest_hashes["synthetic_clean_ledger.csv"], recomputed)
        self.assertNotEqual(digest, "0" * 64)

    def test_empty_output_hashes_rejected(self):
        from test_manifest_schema import valid_manifest
        m = valid_manifest()
        m["output_hashes"] = {}
        ok, _ = P.validate_manifest(m)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()

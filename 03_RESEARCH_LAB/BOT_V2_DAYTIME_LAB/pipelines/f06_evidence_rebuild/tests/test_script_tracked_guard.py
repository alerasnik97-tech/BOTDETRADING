import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline

P = load_pipeline()


class TestScriptTrackedGuard(unittest.TestCase):
    def test_tracked_simulated_passes(self):
        ok, errs = P.check_script_tracked(
            "scripts/f06_rebuild_pipeline.py",
            tracked_paths=["scripts/f06_rebuild_pipeline.py", "README.md"])
        self.assertTrue(ok, errs)

    def test_untracked_simulated_fails(self):
        ok, errs = P.check_script_tracked(
            "scripts/ghost_generator.py", tracked_paths=["README.md"])
        self.assertFalse(ok)
        self.assertTrue(errs)

    def test_empty_tracked_set_fails_closed(self):
        ok, _ = P.check_script_tracked("scripts/x.py", tracked_paths=[])
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()

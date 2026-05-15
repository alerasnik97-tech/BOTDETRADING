import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline

P = load_pipeline()


class TestOutputDirMustNotExist(unittest.TestCase):
    def test_existing_dir_fails_closed(self):
        # the tests directory itself certainly exists
        here = os.path.dirname(os.path.abspath(__file__))
        ok, errs = P.check_output_dir_absent(here)
        self.assertFalse(ok)
        self.assertTrue(any("already exists" in e for e in errs))

    def test_nonexistent_dir_passes(self):
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "future_run_dir_does_not_exist")
            ok, errs = P.check_output_dir_absent(target)
            self.assertTrue(ok, errs)

    def test_empty_path_fails_closed(self):
        ok, _ = P.check_output_dir_absent("")
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()

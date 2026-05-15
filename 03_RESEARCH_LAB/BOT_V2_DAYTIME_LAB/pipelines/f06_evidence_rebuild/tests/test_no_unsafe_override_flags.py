import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import ROOT

SCRIPTS = [
    os.path.join(ROOT, "scripts", "f06_rebuild_pipeline.py"),
    os.path.join(ROOT, "scripts", "validate_rebuild_outputs.py"),
]
# Tokens that would let a caller disable a fail-closed guard.
FORBIDDEN = ["allow_unsafe", "ignore_errors", "skip_guard", "skip_guards",
             "disable_guard", "--force", "force=true", "bypass_guard",
             "no_verify", "--no-verify", "DANGEROUSLY"]


class TestNoUnsafeOverrideFlags(unittest.TestCase):
    def test_no_bypass_tokens_in_source(self):
        for path in SCRIPTS:
            with open(path, "r", encoding="utf-8") as fh:
                src = fh.read().lower()
            for tok in FORBIDDEN:
                self.assertNotIn(tok.lower(), src,
                                 f"unsafe override token '{tok}' in {path}")

    def test_no_argparse_force_like_option(self):
        path = SCRIPTS[0]
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        for opt in ('add_argument("--force"', "add_argument('--force'",
                    'add_argument("--allow-unsafe"', 'add_argument("--skip'):
            self.assertNotIn(opt, src)


if __name__ == "__main__":
    unittest.main()

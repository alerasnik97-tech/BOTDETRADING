import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline

P = load_pipeline()


class TestNoQuarantinedInputs(unittest.TestCase):
    def test_clean_path_passes(self):
        ok, errs = P.check_no_quarantined_path(
            "03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/pipelines/f06_evidence_rebuild/"
            "fixtures/synthetic_clean_ledger.csv")
        self.assertTrue(ok, errs)

    def test_each_quarantine_token_fails(self):
        cases = [
            "reports/QUARANTINED_DO_NOT_USE/x.csv",
            "some/DO_NOT_USE/file.csv",
            "reports/v50b_limited_real_gauntlet_rerun_sw/results/x.csv",
            "trades/V50B_RERUN_TRADES.csv",
            "results/V50B_RERUN_MASTER_RANKING.csv",
        ]
        for c in cases:
            ok, errs = P.check_no_quarantined_path(c)
            self.assertFalse(ok, f"path must be rejected: {c}")
            self.assertTrue(errs)


if __name__ == "__main__":
    unittest.main()

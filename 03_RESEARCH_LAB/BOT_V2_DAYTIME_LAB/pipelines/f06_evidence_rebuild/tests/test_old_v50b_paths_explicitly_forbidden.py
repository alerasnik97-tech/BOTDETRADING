import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline

P = load_pipeline()


class TestOldV50bPathsExplicitlyForbidden(unittest.TestCase):
    def test_named_legacy_artifacts_blocked(self):
        legacy = [
            "03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/"
            "v50b_limited_real_gauntlet_rerun_sw/trades/V50B_RERUN_TRADES.csv",
            "03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/"
            "v50b_limited_real_gauntlet_rerun_sw/results/"
            "V50B_RERUN_MASTER_RANKING.csv",
            "reports/v50b_limited_real_gauntlet_rerun_sw/QUARANTINED_DO_NOT_USE.md",
            "anything/with/DO_NOT_USE/inside.csv",
        ]
        for p in legacy:
            ok, errs = P.check_no_quarantined_path(p)
            self.assertFalse(ok, f"legacy/quarantined path must be blocked: {p}")
            self.assertTrue(errs)

    def test_new_pipeline_paths_allowed(self):
        ok, _ = P.check_no_quarantined_path(
            "03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/pipelines/f06_evidence_rebuild/"
            "fixtures/synthetic_clean_ledger.csv")
        self.assertTrue(ok)

    def test_tokens_constant_covers_both_legacy_csvs(self):
        toks = set(P.QUARANTINE_TOKENS)
        self.assertIn("V50B_RERUN_TRADES.csv", toks)
        self.assertIn("V50B_RERUN_MASTER_RANKING.csv", toks)
        self.assertIn("v50b_limited_real_gauntlet_rerun_sw", toks)


if __name__ == "__main__":
    unittest.main()

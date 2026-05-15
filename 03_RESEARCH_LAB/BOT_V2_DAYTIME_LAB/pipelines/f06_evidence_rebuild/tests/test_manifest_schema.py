import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline

P = load_pipeline()


def valid_manifest():
    return {
        "run_id": "RBdeadbeef", "generated_at": "2026-05-15T00:00:00+00:00",
        "git_branch": "research/f06-evidence-rebuild-foundation-20260515",
        "git_commit_sha": "abc1234", "generator_pid": 123,
        "script_path": "scripts/f06_rebuild_pipeline.py",
        "script_sha256": "a" * 64, "script_is_tracked": True,
        "config_path": "configs/F06_REBUILD_TRAIN_ONLY_TEMPLATE.yaml",
        "config_sha256": "b" * 64, "input_dataset_path": "DRY_RUN_NO_INPUT",
        "input_dataset_sha256_or_reference": "DRY_RUN_NO_INPUT",
        "input_is_quarantined_path": False, "symbol": "EURUSD",
        "families": ["F06"], "exact_months": ["2020-03"],
        "train_only": True, "validation_evaluated": False,
        "holdout_touched": False, "allow_2025": False, "allow_2026": False,
        "row_count_input": 0, "trade_count": 0, "rejected_count": 0,
        "output_hashes": {"DRY_RUN": "c" * 64},
        "safety_flags": {"test_touched": False, "validation_touched": False,
                         "holdout_touched": False, "raw_data_mutated": False,
                         "sweep_run": False, "optimization_run": False},
        "cost_model": {"spread_component": True, "slippage_component": True,
                       "round_turn_commission": True},
        "sample_size_floor": 100, "status": "DRY_RUN_SCHEMA_VALIDATED",
    }


class TestManifestSchema(unittest.TestCase):
    def test_valid_manifest_passes(self):
        ok, errs = P.validate_manifest(valid_manifest())
        self.assertTrue(ok, errs)

    def test_missing_any_required_field_fails(self):
        base = valid_manifest()
        for field in list(base.keys()):
            m = dict(base)
            m.pop(field)
            ok, errs = P.validate_manifest(m)
            self.assertFalse(ok, f"manifest without '{field}' must fail")

    def test_unsafe_const_fails(self):
        for k, bad in (("train_only", False), ("validation_evaluated", True),
                       ("holdout_touched", True), ("allow_2025", True),
                       ("allow_2026", True), ("input_is_quarantined_path", True),
                       ("script_is_tracked", False)):
            m = valid_manifest()
            m[k] = bad
            ok, errs = P.validate_manifest(m)
            self.assertFalse(ok, f"{k}={bad} must fail")

    def test_empty_output_hashes_fails(self):
        m = valid_manifest()
        m["output_hashes"] = {}
        ok, errs = P.validate_manifest(m)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()

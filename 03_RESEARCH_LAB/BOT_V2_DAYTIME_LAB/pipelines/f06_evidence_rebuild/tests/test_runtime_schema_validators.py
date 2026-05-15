import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline
from test_manifest_schema import valid_manifest

P = load_pipeline()


class TestRuntimeSchemaValidators(unittest.TestCase):
    def test_runtime_ledger_schema_requires_run_id(self):
        rows = [{"family_id": "F06", "config_id": "CFG_001",
                 "signal_time": "2024-04-01T10:00:00", "month": "2024-04"}]
        ok, errs = P.validate_ledger_schema(rows)
        self.assertFalse(ok)
        self.assertTrue(any("run_id" in e for e in errs))

    def test_runtime_ledger_schema_requires_datetime_or_month(self):
        rows = [{"run_id": "RBTEST", "family_id": "F06", "config_id": "CFG_001"}]
        ok, errs = P.validate_ledger_schema(rows)
        self.assertFalse(ok)
        self.assertTrue(any("datetime/month/timestamp" in e for e in errs))

    def test_runtime_ranking_schema_rejects_val_columns_train_only(self):
        rows = [{"family_id": "F06", "config_id": "CFG_001", "N_train": "120",
                 "PF_train": "1.4", "Total_R_train": "8.0", "WR_train": "0.51",
                 "N_val": "10"}]
        ok, errs = P.validate_ranking_schema(rows, train_only=True)
        self.assertFalse(ok)
        self.assertTrue(any("validation column" in e for e in errs))

    def test_runtime_ranking_schema_requires_config_id_family_id(self):
        rows = [{"N_train": "120", "PF_train": "1.4",
                 "Total_R_train": "8.0", "WR_train": "0.51"}]
        ok, errs = P.validate_ranking_schema(rows, train_only=True)
        self.assertFalse(ok)
        self.assertTrue(any("family_id" in e for e in errs))
        self.assertTrue(any("config_id" in e for e in errs))

    def test_runtime_cost_schema_requires_spread_slippage_commission(self):
        obj = {"input_ledger_run_id": "RBTEST", "input_is_quarantined_path": False,
               "components_applied": {"slippage_component": True,
                                      "round_turn_commission": True},
               "scenarios": [{"name": "base", "spread_pips": 0.1,
                              "slippage_pips": 0.1,
                              "commission_round_turn_usd": 7.0}] * 3}
        ok, errs = P.validate_cost_report_schema(obj)
        self.assertFalse(ok)
        self.assertTrue(any("spread_component" in e for e in errs))

    def test_runtime_manifest_schema_requires_output_hashes(self):
        m = valid_manifest()
        m.pop("output_hashes")
        ok, errs = P.validate_manifest(m)
        self.assertFalse(ok)
        self.assertTrue(any("output_hashes" in e for e in errs))


if __name__ == "__main__":
    unittest.main()

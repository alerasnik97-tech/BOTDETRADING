import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline, fx

P = load_pipeline()


class TestCostModelComponents(unittest.TestCase):
    def test_all_three_components_pass(self):
        ok, errs = P.check_cost_model_components({
            "spread_component": True, "slippage_component": True,
            "round_turn_commission": True})
        self.assertTrue(ok, errs)

    def test_missing_spread_fails(self):
        ok, errs = P.check_cost_model_components({
            "slippage_component": True, "round_turn_commission": True})
        self.assertFalse(ok)
        self.assertTrue(any("spread" in e for e in errs))

    def test_require_aliases_pass(self):
        ok, errs = P.check_cost_model_components({
            "require_real_spread_component": True,
            "require_slippage_component": True,
            "require_round_turn_commission": True})
        self.assertTrue(ok, errs)

    def test_components_list_pass(self):
        ok, errs = P.check_cost_model_components({
            "components": ["spread_component", "slippage_component",
                           "round_turn_commission"]})
        self.assertTrue(ok, errs)

    def test_empty_fails_closed(self):
        self.assertFalse(P.check_cost_model_components({})[0])

    def test_fixture_has_cost_columns(self):
        header, _ = P.read_csv(fx("synthetic_cost_model_sample.csv"))
        for col in ("spread_pips", "slippage_pips", "commission_round_turn_usd"):
            self.assertIn(col, header)


if __name__ == "__main__":
    unittest.main()

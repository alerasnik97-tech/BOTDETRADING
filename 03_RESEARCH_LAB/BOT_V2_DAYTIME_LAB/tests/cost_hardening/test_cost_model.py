import unittest
import sys
from pathlib import Path

# Add src to sys.path
sys.path.append('.')

from src.v7_engine.cost_model import CostModel, CostModelConfig

class TestCostHardeningModel(unittest.TestCase):
    def test_baseline_preservation(self):
        config = CostModelConfig(mode="zero", slippage_pips=0.0)
        model = CostModel(config)
        res = model.apply_costs_to_trade(gross_r=2.0, sl_pips=20.0)
        self.assertEqual(res["net_r"], 2.0)
        self.assertEqual(res["commission_r"], 0.0)
        self.assertEqual(res["slippage_r"], 0.0)

    def test_slippage_05_reduces_net_r(self):
        config = CostModelConfig(mode="zero", slippage_pips=0.5)
        model = CostModel(config)
        # 0.5 pips over 20 pips SL = 0.5/20 = 0.025 R
        res = model.apply_costs_to_trade(gross_r=2.0, sl_pips=20.0)
        self.assertEqual(res["slippage_r"], 0.025)
        self.assertEqual(res["net_r"], 1.975)

    def test_slippage_10_reduces_net_r_more(self):
        config = CostModelConfig(mode="zero", slippage_pips=1.0)
        model = CostModel(config)
        res = model.apply_costs_to_trade(gross_r=2.0, sl_pips=20.0)
        self.assertEqual(res["slippage_r"], 0.05)
        self.assertEqual(res["net_r"], 1.95)

    def test_ftmo_cost_mode(self):
        # FTMO: $7 per lot. Standard lot = 100k. Pip value = $10.
        # Commission in R = $7 / (SL_pips * $10)
        config = CostModelConfig(mode="ftmo", commission_per_lot_round_turn=7.0, slippage_pips=0.5)
        model = CostModel(config)
        sl_pips = 10.0
        # Comm_R = 7 / (10 * 10) = 0.07 R
        # Slip_R = 0.5 / 10 = 0.05 R
        # Total cost = 0.12 R
        res = model.apply_costs_to_trade(gross_r=1.0, sl_pips=sl_pips)
        self.assertEqual(res["commission_r"], 0.07)
        self.assertEqual(res["slippage_r"], 0.05)
        self.assertEqual(res["net_r"], 0.88)

    def test_stress_combo(self):
        config = CostModelConfig(mode="ftmo", commission_per_lot_round_turn=10.0, slippage_pips=1.0)
        model = CostModel(config)
        sl_pips = 10.0
        # Comm_R = 10 / (10 * 10) = 0.10 R
        # Slip_R = 1.0 / 10 = 0.10 R
        # Total cost = 0.20 R
        res = model.apply_costs_to_trade(gross_r=1.0, sl_pips=sl_pips)
        self.assertEqual(res["net_r"], 0.80)

if __name__ == "__main__":
    unittest.main()

"""Institutional cost-profile routing contract tests.

No backtest / strategy / optimization / sweep / validation / holdout / 2025-26 /
news / high-precision. Pure config+cost-function checks on a fixed synthetic bar.
"""

import unittest

import pandas as pd

from research_lab.config import (
    EngineConfig,
    SUPPORTED_COST_PROFILES,
    SUPPORTED_EXECUTION_MODES,
    resolved_cost_profile,
    with_execution_mode,
)
from research_lab.engine import estimate_spread_pips, estimate_slippage_pips
from research_lab import metric_reconciliation as mr

# A plain mid-session NY bar: no late-session / high-vol multipliers, so the only
# differentiator is the cost profile itself.
TS = pd.Timestamp("2017-06-15 09:30", tz="America/New_York")
RANGE_ATR = 0.5


def _cfg(execution_mode: str, cost_profile: str) -> EngineConfig:
    return with_execution_mode(
        EngineConfig(pair="EURUSD", execution_mode=execution_mode, cost_profile=cost_profile),
        execution_mode,
    )


def _costs(cfg: EngineConfig):
    return (
        estimate_spread_pips("EURUSD", TS, RANGE_ATR, cfg, fill_kind="entry"),
        estimate_slippage_pips(TS, RANGE_ATR, cfg, fill_kind="entry"),
    )


class SupportedEnums(unittest.TestCase):
    def test_supported_profiles_include_three_tiers(self):
        for p in ("base", "conservative", "stress"):
            self.assertIn(p, SUPPORTED_COST_PROFILES)

    def test_supported_modes_include_three_modes(self):
        for m in ("normal_mode", "conservative_mode", "stress_mode"):
            self.assertIn(m, SUPPORTED_EXECUTION_MODES)


class SelfReport(unittest.TestCase):
    def test_1_base_self_reports_base_normal(self):
        cfg = _cfg("normal_mode", "base")
        self.assertEqual(resolved_cost_profile(cfg), "base")
        self.assertEqual(cfg.execution_mode, "normal_mode")

    def test_2_conservative_self_reports_conservative_mode(self):
        cfg = _cfg("conservative_mode", "conservative")
        self.assertEqual(resolved_cost_profile(cfg), "conservative")
        self.assertEqual(cfg.execution_mode, "conservative_mode")

    def test_3_stress_self_reports_stress_mode(self):
        cfg = _cfg("stress_mode", "stress")
        self.assertEqual(resolved_cost_profile(cfg), "stress")
        self.assertEqual(cfg.execution_mode, "stress_mode")

    def test_auto_conservative_mode_maps_to_conservative_not_stress(self):
        # regression: the original mislabel was conservative_mode -> "stress"
        cfg = with_execution_mode(EngineConfig(pair="EURUSD"), "conservative_mode")
        self.assertEqual(resolved_cost_profile(cfg), "conservative")
        cfg2 = with_execution_mode(EngineConfig(pair="EURUSD"), "stress_mode")
        self.assertEqual(resolved_cost_profile(cfg2), "stress")


class Monotonicity(unittest.TestCase):
    def test_5_costs_increase_base_le_conservative_le_stress(self):
        b = _costs(_cfg("normal_mode", "base"))
        c = _costs(_cfg("conservative_mode", "conservative"))
        s = _costs(_cfg("stress_mode", "stress"))
        # spread
        self.assertLess(b[0], c[0])
        self.assertLess(c[0], s[0])
        # slippage
        self.assertLess(b[1], c[1])
        self.assertLess(c[1], s[1])
        # never reduce vs base
        self.assertGreaterEqual(c[0], b[0])
        self.assertGreaterEqual(s[1], b[1])

    def test_4_conservative_and_stress_not_duplicates(self):
        c = _costs(_cfg("conservative_mode", "conservative"))
        s = _costs(_cfg("stress_mode", "stress"))
        self.assertNotEqual(c, s)


class ReconciliationGate(unittest.TestCase):
    def _profiles(self, conservative_cp, stress_cp, stress_mode):
        return {
            "base": {"cost_profile": "base", "execution_mode": "normal_mode"},
            "conservative": {"cost_profile": conservative_cp, "execution_mode": "conservative_mode"},
            "stress": {"cost_profile": stress_cp, "execution_mode": stress_mode},
        }

    def test_7_gate_detects_mislabel(self):
        # conservative folder self-reporting 'stress' (the original defect)
        v = mr.reconcile_cost_profiles(self._profiles("stress", "stress", "conservative_mode"))
        self.assertIn("COST_PROFILE_MISLABEL", {x["code"] for x in v})

    def test_8_gate_detects_duplicate(self):
        v = mr.reconcile_cost_profiles(self._profiles("stress", "stress", "conservative_mode"))
        self.assertIn("COST_PROFILE_DUPLICATE", {x["code"] for x in v})

    def test_6_fixed_routing_passes_gate(self):
        # post-fix mapping: each folder self-reports its own profile, all distinct
        v = mr.reconcile_cost_profiles(self._profiles("conservative", "stress", "stress_mode"))
        self.assertEqual(v, [], f"unexpected violations: {v}")


if __name__ == "__main__":
    unittest.main()

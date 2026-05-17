"""Contract tests for the official formal train runner.

NO backtest / strategy run / dataset / holdout / 2025-26 / optimization /
sweep / validation / news / high-precision. Pure config + policy + gate checks.
"""

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path

from research_lab.config import EngineConfig, resolved_cost_profile
from research_lab.runners import formal_train_runner as R

REPO = Path(__file__).resolve().parents[3]
GOOD_OUT = "03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01/RUN_X"


def _req(**kw):
    base = dict(
        strategy_name="tp01_london_ny_momentum_pullback",
        start_date="2015-01-01",
        end_date="2024-12-31",
        data_path="05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared",
        output_dir=GOOD_OUT,
    )
    base.update(kw)
    return R.FormalRunRequest(**base)


class ImportSafety(unittest.TestCase):
    def test_1_runner_import_has_no_heavy_side_effects(self):
        # Subprocess: importing the runner must NOT import engine/data/strategies.
        env = dict(os.environ, PYTHONPATH=str(REPO / "03_RESEARCH_LAB"))
        code = (
            "import sys; import research_lab.runners.formal_train_runner as m;"
            "assert 'research_lab.engine' not in sys.modules, 'engine imported on import';"
            "assert 'research_lab.data_loader' not in sys.modules;"
            "assert 'research_lab.strategies' not in sys.modules;"
            "print('OK')"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env, cwd=str(REPO))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("OK", out.stdout)

    def test_1b_no_toplevel_heavy_imports_and_has_main_guard(self):
        src = Path(R.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        heavy = {"research_lab.engine", "research_lab.data_loader",
                 "research_lab.strategies", "research_lab.report"}
        for node in tree.body:  # module top level only
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, heavy,
                                 f"{node.module} must be lazy-imported")
        self.assertIn('if __name__ == "__main__":', src)


class CostProfileWiring(unittest.TestCase):
    def setUp(self):
        self.cfgs = R.build_cost_profile_configs(EngineConfig(pair="EURUSD"))

    def test_2_exactly_three_profiles(self):
        self.assertEqual(set(self.cfgs), {"base", "conservative", "stress"})

    def test_3_base_self_reports(self):
        c = self.cfgs["base"]
        self.assertEqual((resolved_cost_profile(c), c.execution_mode), ("base", "normal_mode"))

    def test_4_conservative_self_reports(self):
        c = self.cfgs["conservative"]
        self.assertEqual((resolved_cost_profile(c), c.execution_mode),
                         ("conservative", "conservative_mode"))

    def test_5_stress_self_reports(self):
        c = self.cfgs["stress"]
        self.assertEqual((resolved_cost_profile(c), c.execution_mode), ("stress", "stress_mode"))

    def test_6_monotonicity_validates(self):
        R.validate_cost_profile_configs(self.cfgs)  # must not raise

    def test_7_duplicate_profiles_rejected(self):
        dup = dict(self.cfgs)
        dup["stress"] = self.cfgs["conservative"]  # stress now identical to conservative
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_cost_profile_configs(dup)

    def test_8_mislabeled_profiles_rejected(self):
        import dataclasses
        bad = dict(self.cfgs)
        # force conservative folder to a base/normal config -> resolved != name
        bad["conservative"] = dataclasses.replace(
            self.cfgs["base"], execution_mode="normal_mode", cost_profile="base")
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_cost_profile_configs(bad)


class OutputPolicy(unittest.TestCase):
    def test_9_output_outside_reports_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_output_policy("results/research_lab_robust/run1")

    def test_10_zip_output_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_output_policy(GOOD_OUT + ".zip")

    def test_9b_data_vault_output_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_output_policy("05_MARKET_DATA_VAULT/eurusd_data/x")

    def test_9c_scratch_output_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_output_policy("03_RESEARCH_LAB/scratch/x")

    def test_good_output_accepted(self):
        R.validate_output_policy(GOOD_OUT)  # must not raise

    def test_19_heavy_outputs_route_to_local_outputs(self):
        h = R.heavy_output_dir(GOOD_OUT, "base")
        self.assertIn("local_outputs_do_not_commit", h)
        self.assertTrue(h.endswith("local_outputs_do_not_commit/base"))


class RequestSafety(unittest.TestCase):
    def test_11_holdout_path_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.assert_safe_request(_req(data_path="05_MARKET_DATA_VAULT/eurusd_data/sealed_holdout_2025_2026/prepared"))

    def test_12_future_dates_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.assert_safe_request(_req(end_date="2025-06-01"))
        with self.assertRaises(R.RunnerSafetyError):
            R.assert_safe_request(_req(start_date="2026-01-01", end_date="2026-12-31"))

    def test_13_validation_flag_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.assert_safe_request(_req(validation=True))

    def test_14_optimization_and_sweep_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.assert_safe_request(_req(optimization=True))
        with self.assertRaises(R.RunnerSafetyError):
            R.assert_safe_request(_req(sweep=True))

    def test_15_high_precision_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.assert_safe_request(_req(high_precision=True))

    def test_15b_news_and_non_train_only_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.assert_safe_request(_req(news=True))
        with self.assertRaises(R.RunnerSafetyError):
            R.assert_safe_request(_req(train_only=False))

    def test_safe_request_ok(self):
        R.assert_safe_request(_req())  # must not raise


class DryRunAndSealing(unittest.TestCase):
    def test_16_dry_run_does_not_execute(self):
        plan = R.run_single_strategy_formal_train_only(_req(execute=False))
        self.assertEqual(plan["mode"], "dry_run")
        self.assertFalse(plan["executed"])
        self.assertEqual(set(plan["profiles"]), {"base", "conservative", "stress"})

    def test_16b_preflight_needs_no_dataset(self):
        # preflight must succeed without touching any data file
        plan = R.preflight(_req())
        self.assertFalse(plan["executed"])
        self.assertFalse(plan["manifest"]["holdout_used"])
        self.assertFalse(plan["manifest"]["optimization_run"])

    def test_17_reconciliation_violations_block_sealing(self):
        with self.assertRaises(R.ReconciliationGateError):
            R.seal_run_only_if_reconciled([{"code": "SUMMARY_SELF_CONTRADICTION", "detail": "x"}])

    def test_18_clean_synthetic_ledger_can_seal(self):
        trades = [
            {"direction": "long", "entry_price": 1.10, "exit_price": 1.11,
             "exit_reason": "take_profit", "pnl_usd": 100.0, "pnl_r": 1.0, "result": "win"},
            {"direction": "short", "entry_price": 1.10, "exit_price": 1.11,
             "exit_reason": "stop_loss", "pnl_usd": -100.0, "pnl_r": -1.0, "result": "loss"},
        ]
        eq = [100_000.0, 100_100.0, 100_000.0]
        v = R.reconcile_profile_outputs(trades=trades, equity_series=eq,
                                        starting_equity=100_000.0)
        R.seal_run_only_if_reconciled(v)  # must not raise

    def test_20_manifest_records_only_profiles_run(self):
        m = R.build_run_manifest("tp01", "2015-01-01", "2024-12-31", "p", ["base", "conservative"])
        self.assertEqual(m["profiles_run"], ["base", "conservative"])
        self.assertFalse(m["holdout_used"])
        with self.assertRaises(R.RunnerSafetyError):
            R.build_run_manifest("tp01", "2015-01-01", "2024-12-31", "p", ["base", "base"])


if __name__ == "__main__":
    unittest.main()

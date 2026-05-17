"""Contract tests for the official formal train runner (hardened).

NO backtest / strategy run / dataset / holdout / 2025-26 / optimization /
sweep / validation / news / high-precision. Pure config + scope + policy + gate.
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
GOOD_DATA = "05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared"


def _req(**kw):
    base = dict(
        strategy_name="tp01_london_ny_momentum_pullback",
        start_date="2015-01-01", end_date="2024-12-31",
        data_path=GOOD_DATA, output_dir=GOOD_OUT,
    )
    base.update(kw)
    return R.FormalRunRequest(**base)


class T01_ImportSafety(unittest.TestCase):
    def test_01_import_no_side_effects(self):
        env = dict(os.environ, PYTHONPATH=str(REPO / "03_RESEARCH_LAB"))
        code = ("import sys, research_lab.runners.formal_train_runner;"
                "assert 'research_lab.engine' not in sys.modules;"
                "assert 'research_lab.data_loader' not in sys.modules;"
                "assert 'research_lab.strategies' not in sys.modules;print('OK')")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env, cwd=str(REPO))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("OK", out.stdout)

    def test_01b_no_toplevel_heavy_imports_and_main_guard(self):
        src = Path(R.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        heavy = {"research_lab.engine", "research_lab.data_loader",
                 "research_lab.strategies", "research_lab.report"}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, heavy)
        self.assertIn('if __name__ == "__main__":', src)


class T02_DryRun(unittest.TestCase):
    def test_02_dry_run_default_does_not_execute(self):
        plan = R.run_single_strategy_formal_train_only(_req(execute=False))
        self.assertEqual(plan["mode"], "dry_run")
        self.assertFalse(plan["executed"])
        self.assertTrue(plan["manifest"]["reconciliation_required"])


class T03_CostProfiles(unittest.TestCase):
    def setUp(self):
        self.c = R.build_cost_profile_configs(EngineConfig(pair="EURUSD"))

    def test_03_exactly_three(self):
        self.assertEqual(set(self.c), {"base", "conservative", "stress"})

    def test_04_base_self_reports(self):
        self.assertEqual((resolved_cost_profile(self.c["base"]), self.c["base"].execution_mode),
                         ("base", "normal_mode"))

    def test_05_conservative_self_reports(self):
        self.assertEqual(
            (resolved_cost_profile(self.c["conservative"]), self.c["conservative"].execution_mode),
            ("conservative", "conservative_mode"))

    def test_06_stress_self_reports(self):
        self.assertEqual(
            (resolved_cost_profile(self.c["stress"]), self.c["stress"].execution_mode),
            ("stress", "stress_mode"))

    def test_07_monotonicity_validates(self):
        R.validate_cost_profile_configs(self.c)

    def test_08_duplicate_rejected(self):
        d = dict(self.c); d["stress"] = self.c["conservative"]
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_cost_profile_configs(d)

    def test_09_mislabeled_rejected(self):
        import dataclasses
        d = dict(self.c)
        d["conservative"] = dataclasses.replace(
            self.c["base"], execution_mode="normal_mode", cost_profile="base")
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_cost_profile_configs(d)

    def test_10_high_precision_and_precision_rejected(self):
        import dataclasses
        d = dict(self.c)
        d["stress"] = dataclasses.replace(
            self.c["base"], execution_mode="high_precision_mode", cost_profile="precision")
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_cost_profile_configs(d)


class T11_Scope(unittest.TestCase):
    def test_11_holdout_path_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_train_only_scope("05_MARKET_DATA_VAULT/eurusd_data/holdout/prepared",
                                        "2015-01-01", "2024-12-31")

    def test_12_sealed_holdout_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_train_only_scope(
                "05_MARKET_DATA_VAULT/eurusd_data/sealed_holdout_2025_2026/prepared",
                "2015-01-01", "2024-12-31")

    def test_13_validation_path_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_train_only_scope(
                "05_MARKET_DATA_VAULT/eurusd_data/validation_set/prepared",
                "2015-01-01", "2024-12-31")

    def test_13b_non_train_data_path_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_train_only_scope("05_MARKET_DATA_VAULT/eurusd_data/other/prepared",
                                        "2015-01-01", "2024-12-31")

    def test_14_future_dates_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_train_only_scope(GOOD_DATA, "2015-01-01", "2025-06-01")
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_train_only_scope(GOOD_DATA, "2026-01-01", "2026-12-31")

    def test_scope_ok(self):
        R.validate_train_only_scope(GOOD_DATA, "2015-01-01", "2024-12-31")


class T15_RequestFlags(unittest.TestCase):
    def test_15_optimization_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.assert_safe_request(_req(optimization=True))

    def test_16_sweep_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.assert_safe_request(_req(sweep=True))

    def test_17_validation_flag_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.assert_safe_request(_req(validation=True))

    def test_17b_holdout_news_highprec_nontrain_rejected(self):
        for kw in (dict(holdout=True), dict(news=True),
                   dict(high_precision=True), dict(train_only=False)):
            with self.assertRaises(R.RunnerSafetyError):
                R.assert_safe_request(_req(**kw))

    def test_safe_request_ok(self):
        R.assert_safe_request(_req())


class T18_OutputPolicy(unittest.TestCase):
    def test_18_project_root_rejected(self):
        for bad in (".", "RUN_X"):
            with self.assertRaises(R.RunnerSafetyError):
                R.validate_output_dir(bad)

    def test_19_data_vault_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_output_dir("05_MARKET_DATA_VAULT/eurusd_data/x")

    def test_20_production_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_output_dir("03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/production/x")

    def test_20b_incubation_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_output_dir("incubation/tp01/run")

    def test_21_zip_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_output_dir(GOOD_OUT + ".zip")

    def test_22_scratch_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_output_dir("03_RESEARCH_LAB/scratch/x")

    def test_outside_reports_area_rejected(self):
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_output_dir("results/research_lab_robust/run1")

    def test_good_output_ok(self):
        R.validate_output_dir(GOOD_OUT)

    def test_23_heavy_outputs_routed(self):
        h = R.heavy_output_dir(GOOD_OUT, "base")
        self.assertTrue(h.endswith("local_outputs_do_not_commit/base"))


class T24_Sealing(unittest.TestCase):
    def test_24_violations_block_sealing(self):
        recs = [{"profile": "base", "violations": [{"code": "DRAWDOWN_DEAD"}], "passed": False}]
        with self.assertRaises(R.ReconciliationGateError):
            R.seal_run_only_if_reconciled(recs, {"run_id": "x"})

    def test_24b_missing_manifest_blocks(self):
        with self.assertRaises(R.ReconciliationGateError):
            R.seal_run_only_if_reconciled([{"profile": "base", "violations": []}], None)

    def test_24c_no_reconciliation_blocks(self):
        with self.assertRaises(R.ReconciliationGateError):
            R.seal_run_only_if_reconciled([], {"run_id": "x"})

    def test_25_clean_synthetic_ledger_seals(self):
        trades = [
            {"direction": "long", "entry_price": 1.10, "exit_price": 1.11,
             "exit_reason": "take_profit", "pnl_usd": 100.0, "pnl_r": 1.0, "result": "win"},
            {"direction": "short", "entry_price": 1.10, "exit_price": 1.11,
             "exit_reason": "stop_loss", "pnl_usd": -100.0, "pnl_r": -1.0, "result": "loss"},
        ]
        rec = R.reconcile_profile_outputs(
            "base", trades=trades, equity_series=[100000.0, 100100.0, 100000.0],
            starting_equity=100000.0)
        self.assertTrue(rec["passed"])
        R.seal_run_only_if_reconciled([rec], {"run_id": "x"})  # must not raise


class T26_Manifest(unittest.TestCase):
    def _m(self, profiles):
        return R.build_run_manifest(
            run_id="R1", branch="b", commit="c", strategy_name="tp01",
            data_path=GOOD_DATA, min_timestamp="2015-01-01",
            max_timestamp="2024-12-31", profiles_run=profiles)

    def test_26_records_only_profiles_run(self):
        m = self._m(["base", "conservative"])
        self.assertEqual(m["profiles_run"], ["base", "conservative"])
        self.assertFalse(m["holdout_used"])
        self.assertFalse(m["optimization_run"])
        self.assertTrue(m["reconciliation_required"])

    def test_27_rejects_duplicate_profile_names(self):
        with self.assertRaises(R.RunnerSafetyError):
            self._m(["base", "base"])

    def test_27b_rejects_empty_profiles(self):
        with self.assertRaises(R.RunnerSafetyError):
            self._m([])


class T28_NoDataNoStrategy(unittest.TestCase):
    def test_28_test_module_does_no_file_io(self):
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                self.assertNotIn(name, {"read_csv", "read_parquet", "load_backtest_data_bundle"})

    def test_29_no_2025_2026_used_as_data(self):
        # 2025/2026 appears ONLY as rejected inputs; scope fn is pure (no IO).
        self.assertEqual(R.FIRST_FORBIDDEN_DATE.isoformat(), "2025-01-01")
        with self.assertRaises(R.RunnerSafetyError):
            R.validate_train_only_scope(GOOD_DATA, "2025-01-01", "2025-12-31")

    def test_30_no_strategy_run_required(self):
        # The full contract runs without importing engine/strategies (subprocess).
        env = dict(os.environ, PYTHONPATH=str(REPO / "03_RESEARCH_LAB"))
        code = ("import research_lab.runners.formal_train_runner as R, sys;"
                "R.preflight(R.FormalRunRequest("
                "strategy_name='tp01', start_date='2015-01-01', end_date='2024-12-31',"
                f"data_path={GOOD_DATA!r}, output_dir={GOOD_OUT!r}));"
                "assert 'research_lab.engine' not in sys.modules;"
                "assert 'research_lab.strategies' not in sys.modules;print('OK')")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env, cwd=str(REPO))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("OK", out.stdout)


if __name__ == "__main__":
    unittest.main()

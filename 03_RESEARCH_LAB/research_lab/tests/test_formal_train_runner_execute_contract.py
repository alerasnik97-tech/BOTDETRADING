"""Execute-path contract tests for the official formal train runner.

`execute=True` is driven EXCLUSIVELY through fakes / sys.modules injection:
NO real backtest, NO real market data, NO holdout, NO 2025-26, NO optimization /
sweep / validation / news / high-precision. The fake `run_backtest` /
`summarize_result` / `load_backtest_data_bundle` mirror the REAL required
signatures (verified against engine.py:597-609 and report.py:281-293), so any
regression in the runner's call sites raises immediately. Two extra tests lock
the REAL signatures via `inspect.signature` to guard against engine/report drift.
"""

import contextlib
import inspect
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research_lab.config import EngineConfig
from research_lab.runners import formal_train_runner as R

REPO = Path(R.__file__).resolve().parents[3]
GOOD_DATA = "05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared"
RUN_SUFFIX = "03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/test_run/RUN1"
FAKE_BRANCH = "fix/formal-runner-execute-path-20260517"
FAKE_COMMIT = "deadbeefcafe1234567890abcdef0123456789ab"
_SENTINEL = object()


def _clean_trades_df():
    # Net-zero, sign-consistent ledger: passes every reconcile_* invariant.
    return pd.DataFrame([
        {"direction": "long", "entry_price": 1.10, "exit_price": 1.11,
         "exit_reason": "take_profit", "pnl_usd": 100.0, "pnl_r": 1.0, "result": "win"},
        {"direction": "short", "entry_price": 1.10, "exit_price": 1.11,
         "exit_reason": "stop_loss", "pnl_usd": -100.0, "pnl_r": -1.0, "result": "loss"},
    ])


def _clean_equity_df():
    return pd.DataFrame({
        "datetime_ny": ["2015-01-02 00:00:00", "2018-06-01 00:00:00", "2024-12-31 00:00:00"],
        "equity": [100000.0, 100000.0, 100000.0],
        "drawdown_pct": [0.0, 0.0, 0.0],
    })


def _clean_summary():
    return {"profit_factor": 1.0, "expectancy_r": 0.0,
            "total_return_pct": 0.0, "max_drawdown_pct": 0.0}


def _stats_df():
    return pd.DataFrame({"period": ["2015"], "pnl_usd": [0.0], "trades": [0]})


@contextlib.contextmanager
def fake_runtime(*, summary=None, registry_name="tp01_london_ny_momentum_pullback"):
    """Inject fake heavy modules so the execute path never touches real data."""
    capture: dict = {"loader_calls": [], "run_calls": [], "summarize_calls": []}
    frame = pd.DataFrame({"open": [1.0, 2.0, 3.0, 4.0, 5.0]})  # 5 fake bars
    eff_summary = summary if summary is not None else _clean_summary()

    strat = types.ModuleType("research_lab.strategies")
    strat.STRATEGY_REGISTRY = {
        registry_name: types.SimpleNamespace(
            NAME=registry_name, DEFAULT_PARAMS={}, WARMUP_BARS=0),
    }

    dl = types.ModuleType("research_lab.data_loader")

    def load_backtest_data_bundle(pair, data_dirs, start, end, execution_mode,
                                  *, high_precision_dir=None, target_timeframe="M15"):
        capture["loader_calls"].append({
            "pair": pair, "data_dirs": data_dirs, "start": start, "end": end,
            "execution_mode": execution_mode, "target_timeframe": target_timeframe})
        assert isinstance(data_dirs, list), f"data_dirs must be list, got {type(data_dirs)!r}"
        assert data_dirs and all(isinstance(d, Path) for d in data_dirs), \
            f"data_dirs must be list[Path], got {data_dirs!r}"
        assert execution_mode == "normal_mode"
        return types.SimpleNamespace(frame=frame, data_source_used="fake", precision_package=None)

    dl.load_backtest_data_bundle = load_backtest_data_bundle

    eng = types.ModuleType("research_lab.engine")

    def run_backtest(strategy_module, frame, params, engine_config,
                     news_block, news_filter_used):  # mirrors real REQUIRED sig
        capture["run_calls"].append(engine_config.execution_mode)
        assert news_filter_used is False, "news must stay disabled"
        assert news_block.dtype == bool, f"news_block dtype={news_block.dtype}"
        assert len(news_block) == len(frame), "news_block must match frame length"
        assert not news_block.any(), "news_block must be all-False (no news)"
        return types.SimpleNamespace(
            strategy_name=getattr(strategy_module, "NAME", "fake"),
            trades=_clean_trades_df(),
            equity_curve=pd.DataFrame({"timestamp": [0], "equity": [100000.0]}),
            params=params, news_filter_used=news_filter_used)

    eng.run_backtest = run_backtest

    rep = types.ModuleType("research_lab.report")

    def summarize_result(strategy_name, trades, equity_curve, params,
                         news_filter_used, initial_capital, selected_score):  # real REQUIRED sig
        capture["summarize_calls"].append({
            "initial_capital": initial_capital, "selected_score": selected_score,
            "news_filter_used": news_filter_used})
        assert news_filter_used is False
        assert selected_score is None
        return (dict(eff_summary), _clean_trades_df(),
                _stats_df(), _stats_df(), _clean_equity_df())

    rep.summarize_result = summarize_result

    import research_lab as rlpkg
    fakes = (("research_lab.strategies", "strategies", strat),
             ("research_lab.data_loader", "data_loader", dl),
             ("research_lab.engine", "engine", eng),
             ("research_lab.report", "report", rep))
    saved_mod, saved_attr = {}, {}
    for modname, attr, mod in fakes:
        saved_mod[modname] = sys.modules.get(modname, _SENTINEL)
        saved_attr[attr] = getattr(rlpkg, attr, _SENTINEL)
        sys.modules[modname] = mod
        setattr(rlpkg, attr, mod)
    try:
        yield capture
    finally:
        for modname, attr, _ in fakes:
            if saved_mod[modname] is _SENTINEL:
                sys.modules.pop(modname, None)
            else:
                sys.modules[modname] = saved_mod[modname]
            if saved_attr[attr] is _SENTINEL:
                if hasattr(rlpkg, attr):
                    delattr(rlpkg, attr)
            else:
                setattr(rlpkg, attr, saved_attr[attr])


def _req(output_dir, **kw):
    base = dict(
        strategy_name="tp01_london_ny_momentum_pullback",
        start_date="2015-01-01", end_date="2024-12-31",
        data_path=GOOD_DATA, output_dir=output_dir, execute=True,
    )
    base.update(kw)
    return R.FormalRunRequest(**base)


@contextlib.contextmanager
def _fake_git():
    orig = R.get_git_identity
    R.get_git_identity = lambda repo_root=None: (FAKE_BRANCH, FAKE_COMMIT)
    try:
        yield
    finally:
        R.get_git_identity = orig


class T_RealSignatureLock(unittest.TestCase):
    """Locks the REAL callee signatures (no fakes) — guards engine/report drift."""

    def test_real_run_backtest_requires_news_args(self):
        from research_lab.engine import run_backtest
        ps = inspect.signature(run_backtest).parameters
        self.assertIn("news_block", ps)
        self.assertIn("news_filter_used", ps)
        self.assertIs(ps["news_block"].default, inspect.Parameter.empty)
        self.assertIs(ps["news_filter_used"].default, inspect.Parameter.empty)

    def test_real_summarize_result_requires_capital_and_score(self):
        from research_lab.report import summarize_result
        ps = inspect.signature(summarize_result).parameters
        self.assertIn("initial_capital", ps)
        self.assertIn("selected_score", ps)
        self.assertIs(ps["initial_capital"].default, inspect.Parameter.empty)
        self.assertIs(ps["selected_score"].default, inspect.Parameter.empty)


class T_ExecutePathFakes(unittest.TestCase):
    def test_01_run_backtest_signature_satisfied(self):
        with tempfile.TemporaryDirectory() as td, _fake_git():
            out = str(Path(td) / RUN_SUFFIX)
            with fake_runtime() as cap:
                res = R.run_single_strategy_formal_train_only(_req(out))
            self.assertTrue(res["executed"])
            self.assertEqual(cap["run_calls"],
                             ["normal_mode", "conservative_mode", "stress_mode"])

    def test_02_summarize_result_signature_satisfied(self):
        with tempfile.TemporaryDirectory() as td, _fake_git():
            out = str(Path(td) / RUN_SUFFIX)
            with fake_runtime() as cap:
                R.run_single_strategy_formal_train_only(_req(out))
            self.assertEqual(len(cap["summarize_calls"]), 3)
            for c in cap["summarize_calls"]:
                self.assertEqual(c["initial_capital"], 100000.0)
                self.assertIsNone(c["selected_score"])
                self.assertIs(c["news_filter_used"], False)

    def test_03_data_dirs_normalized_to_list_of_path(self):
        with tempfile.TemporaryDirectory() as td, _fake_git():
            out = str(Path(td) / RUN_SUFFIX)
            with fake_runtime() as cap:
                R.run_single_strategy_formal_train_only(_req(out))
            dd = cap["loader_calls"][0]["data_dirs"]
            self.assertIsInstance(dd, list)
            self.assertTrue(all(isinstance(d, Path) for d in dd))
            self.assertEqual(dd, [Path(GOOD_DATA)])

    def test_04_runs_exactly_three_profiles(self):
        with tempfile.TemporaryDirectory() as td, _fake_git():
            out = str(Path(td) / RUN_SUFFIX)
            with fake_runtime() as cap:
                res = R.run_single_strategy_formal_train_only(_req(out))
            self.assertEqual(cap["run_calls"],
                             ["normal_mode", "conservative_mode", "stress_mode"])
            self.assertEqual(set(res["profiles_meta"]),
                             {"base", "conservative", "stress"})

    def test_05_writes_artifacts_under_valid_output(self):
        with tempfile.TemporaryDirectory() as td, _fake_git():
            out = Path(td) / RUN_SUFFIX
            with fake_runtime():
                res = R.run_single_strategy_formal_train_only(_req(str(out)))
            self.assertTrue((out / "manifests" / "RUN_MANIFEST.json").is_file())
            for p in ("base", "conservative", "stress"):
                self.assertTrue((out / "configs" / f"{p}_ENGINE_CONFIG.json").is_file())
                self.assertTrue((out / "profile_reports" / p / "summary.json").is_file())
                self.assertTrue((out / "profile_reports" / p / "tables" / "monthly.csv").is_file())
                self.assertTrue((out / "profile_reports" / p / "tables" / "yearly.csv").is_file())
                heavy = out / "local_outputs_do_not_commit" / p
                self.assertTrue((heavy / "trades.csv").is_file())
                self.assertTrue((heavy / "equity_curve.csv").is_file())
            zips = list(Path(td).rglob("*.zip"))
            self.assertEqual(zips, [], f"no ZIP may be written: {zips}")
            self.assertEqual(Path(res["artifacts"]["manifest"]).resolve(),
                             (out / "manifests" / "RUN_MANIFEST.json").resolve())

    def test_06_reconciliation_failure_blocks_seal_and_writes(self):
        bad = {"profit_factor": 0.5, "expectancy_r": -0.5,
               "total_return_pct": 5.0, "max_drawdown_pct": 0.0}
        with tempfile.TemporaryDirectory() as td, _fake_git():
            out = Path(td) / RUN_SUFFIX
            with fake_runtime(summary=bad):
                with self.assertRaises(R.ReconciliationGateError):
                    R.run_single_strategy_formal_train_only(_req(str(out)))
            self.assertFalse((out / "manifests" / "RUN_MANIFEST.json").exists())
            self.assertEqual(list(Path(td).rglob("*.json")), [])
            self.assertEqual(list(Path(td).rglob("*.csv")), [])

    def test_07_manifest_has_real_branch_commit(self):
        with tempfile.TemporaryDirectory() as td, _fake_git():
            out = str(Path(td) / RUN_SUFFIX)
            with fake_runtime():
                res = R.run_single_strategy_formal_train_only(_req(out))
            m = res["manifest"]
            self.assertEqual(m["branch"], FAKE_BRANCH)
            self.assertEqual(m["commit"], FAKE_COMMIT)
            self.assertFalse(R._is_placeholder(m["branch"]))
            self.assertFalse(R._is_placeholder(m["commit"]))

    def test_08_cost_profile_reconciliation_blocks_mislabel(self):
        manifest = {"run_id": "x", "branch": "b", "commit": "c"}
        good_rec = [{"profile": "base", "violations": [], "passed": True}]
        mislabel = {
            "base": {"cost_profile": "base", "execution_mode": "normal_mode"},
            "conservative": {"cost_profile": "stress", "execution_mode": "conservative_mode"},
            "stress": {"cost_profile": "stress", "execution_mode": "stress_mode"},
        }
        with self.assertRaises(R.ReconciliationGateError):
            R.seal_run_only_if_reconciled(good_rec, manifest, profiles=mislabel)
        clean = {
            "base": {"cost_profile": "base", "execution_mode": "normal_mode"},
            "conservative": {"cost_profile": "conservative", "execution_mode": "conservative_mode"},
            "stress": {"cost_profile": "stress", "execution_mode": "stress_mode"},
        }
        R.seal_run_only_if_reconciled(good_rec, manifest, profiles=clean)  # must not raise

    def test_09_no_real_market_data_loaded(self):
        with tempfile.TemporaryDirectory() as td, _fake_git():
            out = str(Path(td) / RUN_SUFFIX)
            with fake_runtime() as cap:
                res = R.run_single_strategy_formal_train_only(_req(out))
            # Exactly one loader call, served by the in-memory fake (never disk):
            # proof that no real market data / data_loader was used.
            self.assertTrue(res["executed"])
            self.assertEqual(len(cap["loader_calls"]), 1)
            self.assertEqual(cap["loader_calls"][0]["data_dirs"], [Path(GOOD_DATA)])
            self.assertEqual(cap["loader_calls"][0]["execution_mode"], "normal_mode")
            self.assertEqual(cap["loader_calls"][0]["target_timeframe"], "M1")

    def test_10_no_execute_without_execute_true(self):
        with tempfile.TemporaryDirectory() as td, _fake_git():
            out = str(Path(td) / RUN_SUFFIX)
            with fake_runtime() as cap:
                res = R.run_single_strategy_formal_train_only(_req(out, execute=False))
            self.assertEqual(res["mode"], "dry_run")
            self.assertFalse(res["executed"])
            self.assertEqual(cap["run_calls"], [])
            self.assertEqual(cap["summarize_calls"], [])
            self.assertEqual(cap["loader_calls"], [])

    def test_11_git_identity_unavailable_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / RUN_SUFFIX)
            orig = R.get_git_identity

            def _boom(repo_root=None):
                raise R.RunnerSafetyError("git identity unavailable")

            R.get_git_identity = _boom
            try:
                with fake_runtime():
                    with self.assertRaises(R.RunnerSafetyError):
                        R.run_single_strategy_formal_train_only(_req(out))
            finally:
                R.get_git_identity = orig
            self.assertFalse((Path(td) / RUN_SUFFIX / "manifests" / "RUN_MANIFEST.json").exists())


if __name__ == "__main__":
    unittest.main()

"""Contract + reconciliation tests for the shared metric / equity / cost layer.

Runs with NO backtest, strategy run, optimization, sweep, validation, holdout,
news, high-precision, or 2025/2026 data. Pure functions + synthetic fixtures,
plus one read of an already-committed light summary CSV.
"""

import csv
import unittest
from pathlib import Path

from research_lab.engine import directional_pnl_usd
from research_lab import metric_reconciliation as mr

Q = 1.0          # quote_to_usd rate for a USD-quoted synthetic pair
UNITS = 10_000.0  # unsigned units, as the engine computes them


def _trade(direction, entry, exit_, exit_reason, commission=0.0, risk_usd=100.0):
    gross = directional_pnl_usd(direction, entry, exit_, UNITS, Q)
    pnl_usd = gross - commission
    return {
        "direction": direction,
        "entry_price": entry,
        "exit_price": exit_,
        "exit_reason": exit_reason,
        "pnl_usd": pnl_usd,
        "pnl_r": pnl_usd / risk_usd,
        "result": "win" if pnl_usd > 0 else ("loss" if pnl_usd < 0 else "breakeven"),
    }


class PnlSignInvariants(unittest.TestCase):
    def test_1_long_take_profit_is_win(self):
        self.assertGreater(directional_pnl_usd("long", 1.10, 1.12, UNITS, Q), 0)

    def test_2_long_stop_loss_is_loss(self):
        self.assertLess(directional_pnl_usd("long", 1.10, 1.09, UNITS, Q), 0)

    def test_3_short_take_profit_is_win(self):
        # short profits when price FALLS — this was inverted pre-fix
        self.assertGreater(directional_pnl_usd("short", 1.10, 1.08, UNITS, Q), 0)

    def test_4_short_stop_loss_is_loss(self):
        # short loses when price RISES (hits stop) — was recorded as a win pre-fix
        self.assertLess(directional_pnl_usd("short", 1.10, 1.12, UNITS, Q), 0)

    def test_5_forced_exit_sign_follows_net_pnl(self):
        long_win = _trade("long", 1.10, 1.105, "forced_session_close")
        short_win = _trade("short", 1.10, 1.095, "forced_session_close")
        self.assertEqual(long_win["result"], "win")
        self.assertEqual(short_win["result"], "win")
        self.assertEqual(mr.reconcile_trades([long_win, short_win]), [])

    def test_6_costs_worsen_pnl(self):
        gross = directional_pnl_usd("long", 1.10, 1.11, UNITS, Q)
        net = gross - 5.0  # commission
        self.assertLess(net, gross)


class TradeLevelGate(unittest.TestCase):
    def test_clean_ledger_has_no_violations(self):
        trades = [
            _trade("long", 1.10, 1.12, "take_profit"),
            _trade("short", 1.10, 1.08, "take_profit"),
            _trade("long", 1.10, 1.09, "stop_loss"),
            _trade("short", 1.10, 1.12, "stop_loss"),
        ]
        self.assertEqual(mr.reconcile_trades(trades), [])

    def test_inverted_short_is_flagged(self):
        # Simulate the pre-fix bug: a short that hit stop_loss recorded as +pnl/win
        bad = {
            "direction": "short", "entry_price": 1.10, "exit_price": 1.12,
            "exit_reason": "stop_loss", "pnl_usd": +200.0, "pnl_r": +2.0,
            "result": "win",
        }
        codes = {v["code"] for v in mr.reconcile_trades([bad])}
        self.assertIn("STOP_LOSS_POSITIVE_PNL", codes)

    def test_opposite_sign_pnl_r_flagged(self):
        bad = {"direction": "long", "entry_price": 1.0, "exit_price": 1.0,
               "exit_reason": "time_exit", "pnl_usd": -10.0, "pnl_r": +1.0}
        codes = {v["code"] for v in mr.reconcile_trades([bad])}
        self.assertIn("PNL_SIGN_MISMATCH", codes)


class EquityGate(unittest.TestCase):
    def _ledger(self):
        return [
            _trade("long", 1.10, 1.09, "stop_loss"),    # loss
            _trade("short", 1.10, 1.12, "stop_loss"),    # loss
            _trade("long", 1.10, 1.11, "take_profit"),   # win
        ]

    def test_7_equity_additive_reconciles(self):
        trades = self._ledger()
        start = 100_000.0
        eq = [start]
        for t in trades:
            eq.append(eq[-1] + t["pnl_usd"])
        v = mr.reconcile_equity(trades, eq, start,
                                reported_total_return_pct=(sum(t["pnl_usd"] for t in trades) / start) * 100.0)
        self.assertEqual(v, [], f"unexpected: {v}")

    def test_8_dead_drawdown_detected(self):
        # Reproduce the pre-fix shape: net-losing ledger but a monotone-rising equity
        trades = self._ledger()
        start = 100_000.0
        fake_equity = [start + 500.0 * i for i in range(len(trades) + 1)]  # only goes up
        codes = {v["code"] for v in mr.reconcile_equity(trades, fake_equity, start)}
        self.assertIn("DRAWDOWN_DEAD", codes)
        self.assertIn("ENDING_EQUITY_DECOUPLED", codes)

    def test_8b_real_drawdown_passes(self):
        trades = self._ledger()
        start = 100_000.0
        eq = [start]
        for t in trades:
            eq.append(eq[-1] + t["pnl_usd"])
        codes = {v["code"] for v in mr.reconcile_equity(trades, eq, start)}
        self.assertNotIn("DRAWDOWN_DEAD", codes)
        self.assertNotIn("ENDING_EQUITY_DECOUPLED", codes)

    def test_total_return_sign_decoupled_detected(self):
        trades = self._ledger()  # net negative
        start = 100_000.0
        v = mr.reconcile_equity(trades, [], start, reported_total_return_pct=+135.71)
        self.assertIn("TOTAL_RETURN_SIGN_DECOUPLED", {x["code"] for x in v})


class SummaryGate(unittest.TestCase):
    def test_9_summary_self_contradiction_detected(self):
        v = mr.reconcile_summary(profit_factor=0.8964, expectancy_r=-0.0684,
                                 total_return_pct=+135.71)
        self.assertEqual([x["code"] for x in v], ["SUMMARY_SELF_CONTRADICTION"])

    def test_9b_consistent_summary_ok(self):
        v = mr.reconcile_summary(profit_factor=0.8964, expectancy_r=-0.0684,
                                 total_return_pct=-9.35)
        self.assertEqual(v, [])


class CostProfileGate(unittest.TestCase):
    def test_10_mislabel_and_duplicate_detected(self):
        profiles = {
            "base": {"cost_profile": "base", "execution_mode": "normal_mode"},
            "conservative": {"cost_profile": "stress", "execution_mode": "conservative_mode"},
            "stress": {"cost_profile": "stress", "execution_mode": "conservative_mode"},
        }
        codes = {v["code"] for v in mr.reconcile_cost_profiles(profiles)}
        self.assertIn("COST_PROFILE_MISLABEL", codes)
        self.assertIn("COST_PROFILE_DUPLICATE", codes)

    def test_11_distinct_profiles_ok(self):
        profiles = {
            "base": {"cost_profile": "base", "execution_mode": "normal_mode"},
            "conservative": {"cost_profile": "conservative", "execution_mode": "conservative_mode"},
            "stress": {"cost_profile": "stress", "execution_mode": "high_precision_mode"},
        }
        self.assertEqual(mr.reconcile_cost_profiles(profiles), [])


class ExistingArtifactIsSuspect(unittest.TestCase):
    def test_12_committed_tp01_cost_profile_summary_fails_reconciliation(self):
        root = Path(__file__).resolve().parents[3]
        csv_path = (root / "03_RESEARCH_LAB" / "BOT_V2_DAYTIME_LAB" / "reports" /
                    "formal_train_only" / "tp01_london_ny_momentum_pullback" /
                    "TP01_FORMAL_RERUN_20260516_212500" / "tables" /
                    "TP01_COST_PROFILE_SUMMARY.csv")
        if not csv_path.exists():
            self.skipTest("light artifact not present")
        with csv_path.open(encoding="utf-8", errors="replace") as fh:
            rows = list(csv.DictReader(fh))
        self.assertTrue(rows)
        flagged = False
        for r in rows:
            v = mr.reconcile_summary(
                float(r["pf"]), float(r["expectancy"]), float(r["total_return_pct"]))
            if any(x["code"] == "SUMMARY_SELF_CONTRADICTION" for x in v):
                flagged = True
        self.assertTrue(
            flagged,
            "pre-fix TP-01 artifact must be flagged as metric-inconsistent (suspect)")


class GateNeedsNoMarketData(unittest.TestCase):
    def test_13_module_has_no_data_or_io_dependencies(self):
        # AST-level (ignores docstrings/comments): the gate must not import data
        # loaders nor perform any file/network IO -> it cannot touch market data,
        # holdout, or 2025/2026 anything.
        import ast

        tree = ast.parse(Path(mr.__file__).read_text(encoding="utf-8"))
        banned_imports = {"pandas", "numpy", "requests", "urllib", "pathlib", "os"}
        banned_calls = {"open", "read_csv", "read_parquet", "load", "glob"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.assertNotIn(a.name.split(".")[0], banned_imports)
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], banned_imports)
            if isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                self.assertNotIn(name, banned_calls,
                                 f"reconciliation gate must not call {name}()")


if __name__ == "__main__":
    unittest.main()
